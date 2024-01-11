import torch
import sys

from torch.nn.functional import normalize
PI = 3.141592653589793
from einops import (rearrange, reduce, repeat)


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def coordinate_mapping(coordinates, basis, h, w):

    norm_factor = 1 / torch.tensor([w, h])
    norm_factor = norm_factor.to(coordinates.device)
    normalized_coordinates = coordinates * norm_factor[None, None, ...]
    coordinate_embedding = 2 * PI * normalized_coordinates[..., None] @ basis
    coordinate_embedding = normalize(torch.cat([torch.sin(coordinate_embedding), torch.cos(coordinate_embedding)], dim=-1), p=2.0, dim=3)
    coordinate_embedding = rearrange(coordinate_embedding, 'B K c d -> B K (c d)')

    return coordinate_embedding


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



def flow_loss_func2(flow_preds, correspondence_preds, embedding_preds_A, embedding_preds_B, window_preds, basis, flow_gt, valid,
                   gamma=0.9,
                   max_flow=400,
                   **kwargs,
                   ):
    n_predictions = len(flow_preds)

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid >= 0.5) & (mag < max_flow)

    b, _, h, w = flow_gt.shape
    init_grid = coords_grid(b, h, w).to(flow_gt.device)  # [B, 2, H, W]

    correspondence_gt = init_grid + flow_gt
    correspondence_gt_embedding = rearrange(coordinate_mapping(rearrange(correspondence_gt, 'B C H W -> B (H W) C'), basis, h, w), 'B (H W) C -> B C H W', H=h, W=w)
    
    embedding_loss = []
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        if i == 0:
            correspondence_gt_embedding = rearrange(coordinate_mapping(rearrange(correspondence_gt, 'B C H W -> B (H W) C'), basis, h, w), 'B (H W) C -> B C H W', H=h, W=w)
            i_embedding_loss = ((2.0 - torch.sum(embedding_preds_A[i] * correspondence_gt_embedding, dim=1)) + (2.0 - torch.sum(embedding_preds_B[i] * correspondence_gt_embedding, dim=1))) / 4.0
            embedding_loss.append(i_weight * (valid[:, None] * i_embedding_loss).mean())
        else:
            flow_gt = correspondence_gt - correspondence_preds[i-1]
            flow_gt_embedding = rearrange(coordinate_mapping(rearrange(flow_gt + window_preds[i-1][0], 'B C H W -> B (H W) C'), basis, window_preds[i-1][1], window_preds[i-1][2]), 'B (H W) C -> B C H W', H=h, W=w)
            i_embedding_loss = ((2.0 - torch.sum(embedding_preds_A[i] * flow_gt_embedding, dim=1)) + (2.0 - torch.sum(embedding_preds_B[i] * flow_gt_embedding, dim=1))) / 4.0
            embedding_loss.append(i_weight * 0.001 * (valid[:, None] * i_embedding_loss).mean())

    flow_loss = sum(embedding_loss)

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
        'embedding_loss0': embedding_loss[0].item(),
        'embedding_loss1': embedding_loss[1].item(),
    }

    return flow_loss, metrics, embedding_loss