import math
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from .attention_utils import AttentionAggregator


def get_spherical_weights(
    theta: torch.Tensor,
    phi: torch.Tensor,
    active_mask: torch.Tensor,
    editing_scale: Union[float, List[float]],
    device: torch.device,
    n_attr: int,
) -> torch.Tensor:
    if isinstance(editing_scale, (int, float)):
        editing_scale = [editing_scale] * n_attr

    active_count = int(active_mask.sum().item())

    if active_count == 1:
        editing_scale = [7.5] * n_attr
        theta = torch.tensor(0.0 if active_mask[0] else math.pi / 2, device=device)
        if n_attr > 2 and not active_mask[0]:
            phi = torch.tensor(0.0 if active_mask[1] else math.pi / 2, device=device)

    weights = torch.zeros(n_attr, device=device)

    if active_mask[0]:
        weights[0] = editing_scale[0] * torch.cos(theta)

    if active_mask[1] or (n_attr > 2 and active_mask[2]):
        group_weight = torch.sin(theta)

        if active_mask[1] and (n_attr > 2 and active_mask[2]):
            weights[1] = editing_scale[1] * group_weight * torch.cos(phi)
            weights[2] = editing_scale[2] * group_weight * torch.sin(phi)
        elif active_mask[1]:
            weights[1] = editing_scale[1] * group_weight
        else:
            weights[2] = editing_scale[2] * group_weight

    return weights


def compute_edit_direction(
    unet,
    latents: torch.Tensor,
    t: torch.Tensor,
    uncond_hidden_states: torch.Tensor,
    edit_hidden_states: torch.Tensor,
    cross_attention_kwargs: Optional[Dict[str, Any]],
) -> torch.Tensor:
    noise_pred_edit = unet(
        torch.cat([latents] * 2),
        t,
        encoder_hidden_states=torch.cat([uncond_hidden_states, edit_hidden_states]),
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=False,
    )[0]

    uncond_noise_pred, edit_noise_pred = noise_pred_edit.chunk(2)
    return edit_noise_pred - uncond_noise_pred


def compute_attention_mask(
    grabber,
    target_hw: int = 64,
    batch_cfg: int = 2,
    n_heads: int = 8,
    token_index: int = 1,
) -> torch.Tensor:
    all_attention = list(grabber.all_probs().values())

    aggregator = AttentionAggregator(
        batch_cfg=batch_cfg,
        n_heads=n_heads,
        target_hw=target_hw,
        pool="mean",
    )
    for raw in all_attention:
        aggregator.add_raw(raw)

    global_heat = aggregator.compute()
    mask_token = global_heat[token_index]
    mask = (mask_token - mask_token.min()) / (mask_token.max() - mask_token.min() + 1e-6)
    return mask


def orthogonalize_direction(
    direction: torch.Tensor,
    prev_direction: torch.Tensor,
    current_mask: torch.Tensor,
    prev_mask: torch.Tensor,
    thresh: float,
) -> torch.Tensor:
    threshold_curr = torch.quantile(current_mask, thresh)
    threshold_prev = torch.quantile(prev_mask, thresh)

    current_bin = (current_mask > threshold_curr).float()
    prev_bin = (prev_mask > threshold_prev).float()
    spatial_mask = current_bin * prev_bin

    projection = torch.sum(
        direction * prev_direction * spatial_mask,
        dim=(1, 2, 3),
        keepdim=True,
    )

    direction = direction - projection * prev_direction * spatial_mask
    return F.normalize(direction, dim=1)


def build_active_grid(
    timesteps,
    n_attr: int,
    start_step: List[int],
    end_step: List[int],
    device: torch.device,
) -> torch.Tensor:
    active_grid = torch.zeros((len(timesteps), n_attr), device=device, dtype=torch.bool)
    for j in range(n_attr):
        active_grid[start_step[j] : end_step[j] + 1, j] = True
    return active_grid