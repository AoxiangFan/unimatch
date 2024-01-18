import torch

import torch.nn.functional as F
from torch.nn.functional import normalize
PI = 3.141592653589793
from einops import (rearrange, reduce, repeat)

import sys
sys.path.append("../unimatch")
from unimatch.geometry import coords_grid, coordinate_mapping

def flow_loss_func(flow_preds, flow_gt, valid,
                   gamma=0.9,
                   max_flow=400,
                   **kwargs,
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        i_loss = (flow_preds[i] - flow_gt).abs()

        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss, metrics


def flow_loss_func2(flow_preds, embedding_preds, norms, flow_intermediate, basis, flow_gt, valid,
                   gamma=0.9,
                   max_flow=400,
                   **kwargs,
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    valid_ori = valid

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid_ori >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        i_loss = (flow_preds[i] - flow_gt).abs()

        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    b, _, h_ori, w_ori = flow_gt.shape
    downsample_rate = 8
    flow_loss2 = 0.0
    
    for i in range(len(embedding_preds)):

        i_weight = gamma ** (len(embedding_preds) - i - 1)

        down_flow = F.interpolate(flow_gt, size=(h_ori // downsample_rate, w_ori // downsample_rate), mode='bilinear', align_corners=True) / downsample_rate

        down_mag = F.interpolate(mag[:, None], size=(h_ori // downsample_rate, w_ori // downsample_rate), mode='bilinear', align_corners=True)
        down_valid = F.interpolate(valid_ori[:, None], size=(h_ori // downsample_rate, w_ori // downsample_rate), mode='bilinear', align_corners=True)
        down_valid = (down_valid >= 0.5) & (down_mag < max_flow)
        b, _, h, w = down_flow.shape

        if i == 0:
            init_grid = coords_grid(b, h, w).to(flow_gt.device)  # [B, 2, H, W]

            correspondence_gt = init_grid + down_flow
            correspondence_gt_embedding = rearrange(coordinate_mapping(rearrange(correspondence_gt, 'B C H W -> B (H W) C'), basis, h, w), 'B (H W) C -> B C H W', H=h, W=w)

            i_embedding_loss = (2.0 - torch.sum(embedding_preds[i] * correspondence_gt_embedding, dim=1, keepdim=True))

            flow_loss2 += i_weight * (down_valid * i_embedding_loss).mean()
        else:
            subflow = down_flow - flow_intermediate[i-1]
            norm = norms[i]
            subflow_gt_embedding = rearrange(coordinate_mapping(rearrange(subflow + norm[2], 'B C H W -> B (H W) C'), basis, norm[0], norm[1]), 'B (H W) C -> B C H W', H=h, W=w)
            i_embedding_loss = (2.0 - torch.sum(embedding_preds[i] * subflow_gt_embedding, dim=1, keepdim=True))
            flow_loss2 += i_weight * (down_valid * i_embedding_loss).mean()

        downsample_rate = downsample_rate // 2


    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss + flow_loss2, metrics
