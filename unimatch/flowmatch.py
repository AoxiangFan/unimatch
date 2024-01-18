import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer
from .matching import (global_correlation_softmax, global_correlation_softmax2, local_correlation_softmax, local_correlation_softmax2, local_correlation_with_flow,
                       global_correlation_softmax_stereo, local_correlation_softmax_stereo,
                       correlation_softmax_depth)
from .attention import SelfAttnPropagation
from .geometry import flow_warp, compute_flow_with_depth_pose, embedding_decode, coords_grid
from .reg_refine import BasicUpdateBlock
from .utils import normalize_img, feature_add_position, upsample_flow_with_mask

class DecoderMLP(nn.Module):
    def __init__(self, input_dim=128):
        super(DecoderMLP, self).__init__()
        self.MLP = nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.ReLU(inplace=True),
                nn.Linear(input_dim, input_dim // 2), nn.ReLU(inplace=True),
                nn.Linear(input_dim // 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.MLP(x)

class FlowMatch(nn.Module):
    def __init__(self,
                 num_scales=1,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 reg_refine=False,  # optional local regression refinement
                 task='flow',
                 ):
        super(FlowMatch, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine

        # CNN
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(in_channels=feature_channels)

        if not self.reg_refine or task == 'depth':
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
            # thus far, all the learnable parameters are task-agnostic

        if reg_refine:
            # optional task-specific local regression refinement
            self.refine_proj = nn.Conv2d(128, 256, 1)
            self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=2 if task == 'flow' else 1,
                                           bilinear_up=task == 'depth',
                                           )
            
        coordinate_embedding_decoder = torch.load("/scratch/cvlab/home/afan/projects/unimatch/decoder/color_embedding_decoder.pth")
        self.basis = coordinate_embedding_decoder['basis'].to("cuda")
        self.embedding_dimension = coordinate_embedding_decoder['embedding_dimension']
        self.coordinateEmbeddingDecoder = DecoderMLP(input_dim=self.embedding_dimension)
        self.coordinateEmbeddingDecoder.load_state_dict(coordinate_embedding_decoder['model_state_dict'])
        for param in self.coordinateEmbeddingDecoder.parameters():
            param.requires_grad = False

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_flow

    def forward(self, img0, img1,
                attn_type=None,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                num_reg_refine=1,
                pred_bidir_flow=False,
                task='flow',
                intrinsics=None,
                pose=None,  # relative pose transform
                min_depth=1. / 0.5,  # inverse depth range
                max_depth=1. / 10,
                num_depth_candidates=64,
                depth_from_argmax=False,
                pred_bidir_depth=False,
                **kwargs,
                ):

        results_dict = {}
        flow_preds = []
        embedding_preds = []
        norms = []
        flow_intermediate = []

        # stereo and depth tasks have normalized img in dataloader
        img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # list of features, resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2
                flow_intermediate.append(flow.detach())

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1,
                                                  attn_type=attn_type,
                                                  attn_num_splits=attn_splits,
                                                  )

            # correlation and softmax
            if corr_radius == -1:  # global matching
                correspondence_embedding = global_correlation_softmax2(feature0, feature1, self.basis, pred_bidir_flow)[0]
                embedding_preds.append(correspondence_embedding)
                b, c, h, w = feature0.size()
                norms.append((h, w, 0))
                correspondence = embedding_decode(correspondence_embedding, self.coordinateEmbeddingDecoder, self.embedding_dimension, (h, w, 0))
                init_grid = coords_grid(b, h, w).to(correspondence.device) 
                flow_pred = correspondence - init_grid
            else:  # local matching
                flow_embedding, _, norm = local_correlation_softmax2(feature0, feature1, corr_radius, correspondence, self.basis)
                embedding_preds.append(flow_embedding)
                norms.append(norm)
                flow_pred = embedding_decode(flow_embedding, self.coordinateEmbeddingDecoder, self.embedding_dimension, norm)

            flow = flow + flow_pred if flow is not None else flow_pred

            # upsample to the original resolution for supervison at training time only
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                flow_preds.append(flow_bilinear)

            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius,
                                          )
            
            # bilinear exclude the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                # upsample to the original image resolution
                flow_up = self.upsample_flow(flow, feature0)
                flow_preds.append(flow_up)

        results_dict.update({'flow_preds': flow_preds})

        results_dict.update({'embedding_preds': embedding_preds})

        results_dict.update({'norms': norms})

        results_dict.update({'flow_intermediate': flow_intermediate})

        results_dict.update({'basis': self.basis})

        return results_dict
