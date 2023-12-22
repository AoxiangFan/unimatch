import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer
from .matching import (global_correlation_softmax, local_correlation_softmax, local_correlation_with_flow,
                       global_correlation_softmax_stereo, local_correlation_softmax_stereo,
                       correlation_softmax_depth)
from .attention import SelfAttnPropagation
from .geometry import flow_warp, compute_flow_with_depth_pose, coords_grid
from .reg_refine import BasicUpdateBlock
from .utils import normalize_img, feature_add_position, upsample_flow_with_mask

import sys
from einops import (rearrange, reduce, repeat)

class DecoderMLP(nn.Module):
    def __init__(self, input_dim=128):
        super(DecoderMLP, self).__init__()
        self.MLP = nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.ReLU(inplace=True),
                nn.Linear(input_dim, input_dim // 2), nn.ReLU(inplace=True),
                nn.Linear(input_dim // 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.MLP(x)


class UniMatch(nn.Module):
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
        super(UniMatch, self).__init__()

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


        coordinate_embedding_decoder = torch.load("/scratch/cvlab/home/afan/projects/unimatch/checkpoints_saved/coordinate_embedding_decoder.pth")
        self.basis = coordinate_embedding_decoder['basis'].to("cuda")
        self.embedding_dimension = coordinate_embedding_decoder['embedding_dimension']
        self.coordinateEmbeddingDecoder = DecoderMLP(input_dim=self.embedding_dimension)
        self.coordinateEmbeddingDecoder.load_state_dict(coordinate_embedding_decoder['model_state_dict'])
        for param in self.coordinateEmbeddingDecoder.parameters():
            param.requires_grad = False

        if not self.reg_refine or task == 'depth':
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            # self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
            #                                nn.ReLU(inplace=True),
            #                                nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
            
            self.upsampler2 = nn.Sequential(nn.Conv2d(self.embedding_dimension * 2 + feature_channels, 256, 3, 1, 1),
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
    
    def upsample_embedding(self, flow, feature, bilinear=False, upsample_factor=8,
                      is_depth=True):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler2(concat)
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

        if pred_bidir_flow:
            assert task == 'flow'

        if task == 'depth':
            assert self.num_scales == 1  # multi-scale depth model is not supported yet

        results_dict = {}
        flow_preds = []
        correspondence_embedding_preds = []

        if task == 'flow':
            # stereo and depth tasks have normalized img in dataloader
            img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # list of features, resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None
        correspondence = None

        if task != 'depth':
            assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales
        else:
            assert len(attn_splits_list) == len(prop_radius_list) == self.num_scales == 1

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)

            feature0_ori, feature1_ori = feature0, feature1

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if task == 'depth':
                # scale intrinsics
                intrinsics_curr = intrinsics.clone()
                intrinsics_curr[:, :2] = intrinsics_curr[:, :2] / upsample_factor

            if scale_idx > 0:
                assert task != 'depth'  # not supported for multi-scale depth model
                correspondence = F.interpolate(correspondence, scale_factor=2, mode='bilinear', align_corners=True) * 2
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

                

            if flow is not None:
                assert task != 'depth'
                flow = flow.detach()

                if task == 'stereo':
                    # construct flow vector for disparity
                    # flow here is actually disparity
                    zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                    # NOTE: reverse disp, disparity is positive
                    displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                    feature1 = flow_warp(feature1, displace)  # [B, C, H, W]
                elif task == 'flow':
                    feature1 = flow_warp(feature1, flow)  # [B, C, H, W]
                else:
                    raise NotImplementedError

            attn_splits = attn_splits_list[scale_idx]
            if task != 'depth':
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
            if task == 'depth':
                # first generate depth candidates
                b, _, h, w = feature0.size()
                depth_candidates = torch.linspace(min_depth, max_depth, num_depth_candidates).type_as(feature0)
                depth_candidates = depth_candidates.view(1, num_depth_candidates, 1, 1).repeat(b, 1, h,
                                                                                               w)  # [B, D, H, W]

                flow_pred = correlation_softmax_depth(feature0, feature1,
                                                      intrinsics_curr,
                                                      pose,
                                                      depth_candidates=depth_candidates,
                                                      depth_from_argmax=depth_from_argmax,
                                                      pred_bidir_depth=pred_bidir_depth,
                                                      )[0]

            else:
                if corr_radius == -1:  # global matching
                    if task == 'flow':
                        correspondence_embedding, _ = global_correlation_softmax(feature0, feature1, self.basis, pred_bidir_flow)
                    elif task == 'stereo':
                        flow_pred = global_correlation_softmax_stereo(feature0, feature1)[0]
                    else:
                        raise NotImplementedError
                else:  # local matching
                    if task == 'flow':
                        correspondence_embedding, _ = local_correlation_softmax(feature0, feature1, corr_radius, correspondence, self.basis)

                    elif task == 'stereo':
                        flow_pred = local_correlation_softmax_stereo(feature0, feature1, corr_radius)[0]
                    else:
                        raise NotImplementedError

            # flow or residual flow
            # flow = flow + flow_pred if flow is not None else flow_pred
                

            if task == 'stereo':
                flow = flow.clamp(min=0)  # positive disparity

            # upsample to the original resolution for supervison at training time only
            if self.training:
                # flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                #                                    is_depth=task == 'depth')
                # flow_preds.append(flow_bilinear)

                correspondence_embedding_bilinear = self.upsample_embedding(correspondence_embedding, None, bilinear=True, upsample_factor=upsample_factor)
                
                correspondence_embedding_preds.append(correspondence_embedding_bilinear)

            # flow propagation with self-attn
            if (pred_bidir_flow or pred_bidir_depth) and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation

            # flow = self.feature_flow_attn(feature0, flow.detach(),
            #                               local_window_attn=prop_radius > 0,
            #                               local_window_radius=prop_radius,
            #                               )
            
            correspondence_embedding = self.feature_flow_attn(feature0, correspondence_embedding.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius,
                                          )

            # bilinear exclude the last one
            if self.training and scale_idx < self.num_scales - 1:
                # flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                #                              upsample_factor=upsample_factor,
                #                              is_depth=task == 'depth')
                # flow_preds.append(flow_up)

                correspondence_embedding_up = self.upsample_embedding(correspondence_embedding, feature0, bilinear=True,
                                             upsample_factor=upsample_factor)
                correspondence_embedding_preds.append(correspondence_embedding_up)



            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    # upsample to the original image resolution

                    if task == 'stereo':
                        flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                        flow_up_pad = self.upsample_flow(flow_pad, feature0)
                        flow_up = -flow_up_pad[:, :1]  # [B, 1, H, W]
                    elif task == 'depth':
                        depth_pad = torch.cat((flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                        depth_up_pad = self.upsample_flow(depth_pad, feature0,
                                                          is_depth=True).clamp(min=min_depth, max=max_depth)
                        flow_up = depth_up_pad[:, :1]  # [B, 1, H, W]
                    else:
                        # flow_up = self.upsample_flow(flow, feature0)
                        correspondence_embedding_up = self.upsample_embedding(correspondence_embedding, feature0)

                    # flow_preds.append(flow_up)
                    correspondence_embedding_preds.append(correspondence_embedding_up)

                    B, C, H, W = correspondence_embedding_up.shape
                    decoded_correspondence_1 = self.coordinateEmbeddingDecoder(rearrange(correspondence_embedding_up, 'B C H W -> (B H W) C')[:, 0:64])
                    decoded_correspondence_2 = self.coordinateEmbeddingDecoder(rearrange(correspondence_embedding_up, 'B C H W -> (B H W) C')[:, 64:])
                    decoded_correspondence = torch.cat((decoded_correspondence_1, decoded_correspondence_2), dim=1)
                    
                    denorm_factor = torch.tensor([W, H])
                    denorm_factor = denorm_factor.to(decoded_correspondence.device)
                    decoded_correspondence = decoded_correspondence * denorm_factor[None, :]
                    decoded_correspondence = rearrange(decoded_correspondence, '(B H W) C -> B C H W', B=B, H=H, W=W)

                    init_grid = coords_grid(B, H, W)
                    flow_preds.append(decoded_correspondence - init_grid.to(decoded_correspondence.device))


                    
                else:
                    # task-specific local regression refinement
                    # supervise current flow
                    if self.training:
                        flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                     upsample_factor=upsample_factor,
                                                     is_depth=task == 'depth')
                        flow_preds.append(flow_up)

                    assert num_reg_refine > 0
                    for refine_iter_idx in range(num_reg_refine):
                        flow = flow.detach()

                        if task == 'stereo':
                            zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                            # NOTE: reverse disp, disparity is positive
                            displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=displace,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]
                        elif task == 'depth':
                            if pred_bidir_depth and refine_iter_idx == 0:
                                intrinsics_curr = intrinsics_curr.repeat(2, 1, 1)
                                pose = torch.cat((pose, torch.inverse(pose)), dim=0)

                                feature0_ori, feature1_ori = torch.cat((feature0_ori, feature1_ori),
                                                                       dim=0), torch.cat((feature1_ori,
                                                                                          feature0_ori), dim=0)

                            flow_from_depth = compute_flow_with_depth_pose(1. / flow.squeeze(1),
                                                                           intrinsics_curr,
                                                                           extrinsics_rel=pose,
                                                                           )

                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=flow_from_depth,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]

                        else:
                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=flow,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]

                        proj = self.refine_proj(feature0)

                        net, inp = torch.chunk(proj, chunks=2, dim=1)

                        net = torch.tanh(net)
                        inp = torch.relu(inp)

                        net, up_mask, residual_flow = self.refine(net, inp, correlation, flow.clone(),
                                                                  )

                        if task == 'depth':
                            flow = (flow - residual_flow).clamp(min=min_depth, max=max_depth)
                        else:
                            flow = flow + residual_flow

                        if task == 'stereo':
                            flow = flow.clamp(min=0)  # positive

                        if self.training or refine_iter_idx == num_reg_refine - 1:
                            if task == 'depth':
                                if refine_iter_idx < num_reg_refine - 1:
                                    # bilinear upsampling
                                    flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                                 upsample_factor=upsample_factor,
                                                                 is_depth=True)
                                else:
                                    # last one convex upsampling
                                    # NOTE: clamp depth due to the zero padding in the unfold in the convex upsampling
                                    # pad depth to 2 channels as flow
                                    depth_pad = torch.cat((flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                                    depth_up_pad = self.upsample_flow(depth_pad, feature0,
                                                                      is_depth=True).clamp(min=min_depth,
                                                                                           max=max_depth)
                                    flow_up = depth_up_pad[:, :1]  # [B, 1, H, W]

                            else:
                                flow_up = upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor,
                                                                  is_depth=task == 'depth')

                            flow_preds.append(flow_up)

            B, C, H, W = correspondence_embedding.shape
            decoded_correspondence_1 = self.coordinateEmbeddingDecoder(rearrange(correspondence_embedding, 'B C H W -> (B H W) C')[:, 0:64])
            decoded_correspondence_2 = self.coordinateEmbeddingDecoder(rearrange(correspondence_embedding, 'B C H W -> (B H W) C')[:, 64:])
            decoded_correspondence = torch.cat((decoded_correspondence_1, decoded_correspondence_2), dim=1)

            
            denorm_factor = torch.tensor([W, H])
            denorm_factor = denorm_factor.to(decoded_correspondence.device)
            decoded_correspondence = decoded_correspondence * denorm_factor[None, :]
            decoded_correspondence = rearrange(decoded_correspondence, '(B H W) C -> B C H W', B=B, H=H, W=W)

            correspondence = decoded_correspondence.detach()

            init_grid = coords_grid(B, H, W)
            flow = decoded_correspondence - init_grid.to(correspondence.device)



        if task == 'stereo':
            for i in range(len(flow_preds)):
                flow_preds[i] = flow_preds[i].squeeze(1)  # [B, H, W]

        # convert inverse depth to depth
        if task == 'depth':
            for i in range(len(flow_preds)):
                flow_preds[i] = 1. / flow_preds[i].squeeze(1)  # [B, H, W]

        results_dict.update({'flow_preds': flow_preds})

        results_dict.update({'correspondence_embedding_preds': correspondence_embedding_preds})

        results_dict.update({'basis': self.basis})

        return results_dict
