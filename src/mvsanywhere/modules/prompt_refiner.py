"""Prompt refinement head: corrects a predicted depth map with sparse prompts.

DriveMVS addition; not part of the upstream MVSAnywhere code base.

Why this exists. Feeding prompts only into the cost volume and into the ViT
tokens (the design described in the first DriveMVS paper) leaves it to the
network to discover that the prompt is the answer at the prompt pixels. In our
re-training the network ignored the prompts for a long time: after 1k updates
even dense ground truth given as prompt did not change the output. This head
makes the relation explicit. It works on what is easy to use: the *residual*
between each prompt and the network's own preliminary prediction, in the
normalised space the network predicts in.

How it works.
    1. residual = (prompt - preliminary prediction) at prompt pixels, for the
       prompt of the reference view and for the prompt transferred from the
       source views separately.
    2. The sparse residuals are spread with normalised convolutions at cell
       sizes 1, 2, 4, ... pixels: each level holds the mean residual of the
       valid samples in a cell and how many there were. Coarse levels carry
       "the whole region is 20 % too far" (e.g. scale collapse of MVS when the
       vehicle stands still), fine levels carry local detail.
    3. A small convolutional network sees these maps together with decoder
       features and the preliminary prediction and outputs a dense correction.

Without prompts every prompt input is zero, so the head has nothing to copy
from; its last layer starts at zero, so an untrained head leaves the
preliminary prediction unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def residual_pyramid(residual: Tensor, mask: Tensor, num_levels: int) -> Tensor:
    """Sparsity-aware multi-scale spreading of sparse residuals.

    Args:
        residual: (B, 1, H, W), zero where invalid.
        mask: (B, 1, H, W) float validity.
        num_levels: level l averages over cells of 2**l pixels.

    Returns:
        (B, 2 * num_levels, H, W): per level the mean residual of the valid
        samples in the cell and the share of valid samples (square-rooted to
        compress its range), bilinearly upsampled back to (H, W).
    """
    height, width = residual.shape[-2:]
    outputs = []
    for level in range(num_levels):
        cell = 2**level
        if cell == 1:
            mean, density = residual * mask, mask
        else:
            density = F.avg_pool2d(mask, cell, ceil_mode=True)
            mean = F.avg_pool2d(residual * mask, cell, ceil_mode=True) / density.clamp_min(1e-6)
            mean = torch.where(density > 0, mean, torch.zeros_like(mean))
            mean = F.interpolate(mean, size=(height, width), mode="bilinear", align_corners=False)
            density = F.interpolate(density, size=(height, width), mode="bilinear", align_corners=False)
        outputs += [mean, density.sqrt()]
    return torch.cat(outputs, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        y = self.conv1(F.gelu(self.norm1(x)))
        y = self.conv2(F.gelu(self.norm2(y)))
        return x + y


class PromptRefiner(nn.Module):
    """Predicts a dense correction of the preliminary prediction from sparse
    prompt residuals, see the module docstring.

    The correction is mostly a *blend* of the spread residuals: for every
    pixel the network outputs weights over (prompt kind, pyramid level) plus a
    "keep the prediction" option, normalised with a softmax, and the
    correction is the weighted sum of the residual maps. Levels without any
    prompt sample near a pixel cannot be selected. The network therefore only
    has to decide *which* evidence to trust, not to reproduce its value. A
    small free term (starting at zero) can add detail beyond the blend.
    """

    # the residual maps are multiplied by this factor before they enter the
    # network: typical residuals are ~0.1 in logit space and would otherwise
    # drown next to the image features.
    INPUT_GAIN = 10.0

    def __init__(self, feature_channels, hidden=64, num_levels=7, num_prompt_kinds=2, max_residual=8.0, keep_bias=0.0):
        super().__init__()
        self.num_levels = num_levels
        self.num_prompt_kinds = num_prompt_kinds
        self.max_residual = max_residual
        num_maps = num_prompt_kinds * num_levels

        self.feature_proj = nn.Conv2d(feature_channels, 32, 1)
        self.stem = nn.Conv2d(32 + 1 + 2 * num_maps, hidden, 3, padding=1)
        self.blocks = nn.Sequential(
            ResidualBlock(hidden, 1),
            ResidualBlock(hidden, 2),
            ResidualBlock(hidden, 4),
            ResidualBlock(hidden, 1),
        )
        # outputs: one blend logit per residual map, the "keep" logit and the
        # free correction term.
        self.head = nn.Sequential(nn.GroupNorm(8, hidden), nn.GELU(), nn.Conv2d(hidden, num_maps + 2, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        # Start from a sensible fixed rule instead of "do nothing": trust the
        # regional mean residual (cells of 8 px and larger) and leave fine
        # levels for training to open up. On KITTI this rule alone already
        # removes most of the scale error of the preliminary prediction.
        with torch.no_grad():
            level_bias = torch.linspace(-3.0, 3.0, num_levels).clamp(max=1.0)
            self.head[-1].bias[:num_maps] = level_bias.repeat(num_prompt_kinds)
            self.head[-1].bias[num_maps] = keep_bias

    def forward(self, prediction: Tensor, features: Tensor, prompts: Tensor) -> Tensor:
        """
        Args:
            prediction: (B, 1, H, W) preliminary prediction (logit space).
            features: (B, C, h, w) decoder features, any resolution.
            prompts: (B, 2 * num_prompt_kinds, H, W); for each prompt kind its
                value in the same space as `prediction` and its validity mask.

        Returns:
            (B, 1, H, W) refined prediction.
        """
        prediction = prediction.float()
        height, width = prediction.shape[-2:]
        means, densities = [], []
        for kind in range(self.num_prompt_kinds):
            value = prompts[:, 2 * kind : 2 * kind + 1].float()
            mask = prompts[:, 2 * kind + 1 : 2 * kind + 2].float()
            residual = ((value - prediction) * mask).clamp(-self.max_residual, self.max_residual)
            pyramid = residual_pyramid(residual, mask, self.num_levels)
            means.append(pyramid[:, 0::2])
            densities.append(pyramid[:, 1::2])
        means = torch.cat(means, dim=1)  # (B, kinds * levels, H, W)
        densities = torch.cat(densities, dim=1)

        feats = self.feature_proj(features.float())
        feats = F.interpolate(feats, size=(height, width), mode="bilinear", align_corners=False)
        x = torch.cat([feats, prediction.clamp(-10, 10), means * self.INPUT_GAIN, densities], dim=1)
        out = self.head(self.blocks(self.stem(x)))

        num_maps = means.shape[1]
        blend_logits = out[:, :num_maps].masked_fill(densities <= 0, -1e4)
        weights = torch.softmax(torch.cat([blend_logits, out[:, num_maps : num_maps + 1]], dim=1), dim=1)
        correction = (weights[:, :num_maps] * means).sum(dim=1, keepdim=True) + out[:, num_maps + 1 :]
        return prediction + correction
