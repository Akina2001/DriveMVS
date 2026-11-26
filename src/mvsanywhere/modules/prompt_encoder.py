"""Sparsity-aware encoder that turns a sparse depth prompt into ViT tokens.

DriveMVS addition; not part of the upstream MVSAnywhere code base.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvGNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


def masked_max_pool2d(x: Tensor, mask: Tensor, kernel: int = 2, stride: int = 2):
    """Max pooling that only looks at valid positions.

    Ordinary pooling or strided convolutions dilute the few valid samples of a
    sparse map with the zeros around them. Here invalid positions are pushed to
    the smallest representable value before pooling, so a single valid sample
    in a window survives; windows without any valid sample become zero.

    Args:
        x: (B, C, H, W) features.
        mask: (B, 1, H, W) float validity mask.

    Returns:
        pooled features and pooled mask.
    """
    neg_inf = torch.finfo(x.dtype).min
    x_ = x.masked_fill(~mask.bool(), neg_inf)
    x_p = F.max_pool2d(x_, kernel, stride)
    m_p = F.max_pool2d(mask, kernel, stride)
    x_p = torch.where(m_p.bool(), x_p, torch.zeros_like(x_p))
    return x_p, m_p


def build_2d_sincos_pos_embed(d_model: int, h: int, w: int, device) -> Tensor:
    """(h * w, d_model) sinusoidal embedding of normalised 2D positions."""
    assert d_model % 2 == 0
    d_half = d_model // 2

    def sincos_1d(length, dim):
        pos = torch.linspace(0, 1, steps=length, device=device)
        dim_t = torch.arange(dim // 2, device=device) / (dim / 2)
        freq = 1.0 / (10000**dim_t)
        ang = pos[:, None] * freq[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    pe_y = sincos_1d(h, d_half)
    pe_x = sincos_1d(w, d_half)
    pe = torch.cat([pe_y[:, None, :].expand(h, w, -1), pe_x[None, :, :].expand(h, w, -1)], dim=-1)
    return pe.reshape(h * w, d_model)


class SparsePromptPatchEmbed(nn.Module):
    """Encodes a sparse prompt into one token per 16x16 patch.

    The input has two channels, the prompt in the network's prediction space
    and its validity mask (see ``prompt_utils.normalized_log_prompt``).
    Normalised pixel coordinates are appended, then a convolutional stem and
    three stages, each followed by masked max pooling, bring the map to 1/16
    resolution. Patches without any valid sample yield an all-zero token.
    """

    def __init__(self, in_channels=2, embed_dim=768, add_coords=True) -> None:
        super().__init__()

        self.add_coords = add_coords
        in_ch = in_channels + 2 if add_coords else in_channels

        channels = [64, 128, 256, 512]
        self.stem = nn.Sequential(
            ConvGNReLU(in_ch, channels[0]), ConvGNReLU(channels[0], channels[0])
        )
        self.stage1 = nn.ModuleList(
            [ConvGNReLU(channels[i], channels[i + 1]) for i in range(3)]
        )
        self.stage2 = nn.ModuleList(
            [ConvGNReLU(channels[i + 1], channels[i + 1]) for i in range(3)]
        )

        self.proj = nn.Conv2d(channels[-1], embed_dim, kernel_size=1, bias=True)
        self.token_ln = nn.LayerNorm(embed_dim)

    @staticmethod
    def _coords(B, H, W, device, dtype):
        ys = torch.linspace(-1, 1, steps=H, device=device, dtype=dtype).view(1, 1, H, 1)
        xs = torch.linspace(-1, 1, steps=W, device=device, dtype=dtype).view(1, 1, 1, W)
        return ys.expand(B, 1, H, W), xs.expand(B, 1, H, W)

    def forward(self, x: Tensor, mask: Tensor):
        """
        Args:
            x: (B, 2, H, W) encoded prompt and mask; H and W divisible by 16.
            mask: (B, 1, H, W) float validity mask.

        Returns:
            tokens: (B, H/16 * W/16, embed_dim), zero where a patch holds no
                valid prompt sample.
            token_mask: (B, H/16 * W/16) bool, True where a patch holds at
                least one valid sample.
        """
        B, _, H, W = x.shape
        assert H % 16 == 0 and W % 16 == 0, "Prompt size must be divisible by 16."

        if self.add_coords:
            ys, xs = self._coords(B, H, W, x.device, x.dtype)
            x = torch.cat([x, ys, xs], dim=1)

        x = self.stem(x)
        x, m = masked_max_pool2d(x, mask.to(x.dtype))

        for i in range(3):
            x = self.stage2[i](self.stage1[i](x))
            x, m = masked_max_pool2d(x, m)

        x = self.proj(x)
        _, D, Hf, Wf = x.shape
        tokens = self.token_ln(x.flatten(2).transpose(1, 2))
        tokens = tokens + build_2d_sincos_pos_embed(D, Hf, Wf, x.device).to(tokens.dtype).unsqueeze(0)

        token_mask = m.reshape(B, -1) > 0
        tokens = tokens.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        return tokens, token_mask
