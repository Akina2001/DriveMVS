"""Utilities for sparse metric depth prompts (LiDAR).

A *prompt* is a sparse depth map in metres with the same resolution as the
depth the network predicts. A value of 0 means "no measurement"; an explicit
float mask (1 = valid) travels with it under the ``sparse_prompt_mask_b1hw``
key.

Contents
    * simulating LiDAR prompts from dense synthetic depth (training),
    * sub-sampling scan lines of a real LiDAR projection (evaluation),
    * prompt augmentations (noise, point dropout, limited field of view),
    * moving prompts between views with known geometry
      (:func:`transfer_sparse_depth`),
    * the prompt-dropout modes used during training, and
    * small helpers shared by the cost volume and the prompt encoder.

DriveMVS addition; not part of the upstream MVSAnywhere code base.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _scatter_min_depth(
    flat_index: Tensor, depth: Tensor, num_cells: int, fill_value: float = 0.0
) -> Tensor:
    """Scatter ``depth`` into a flat buffer keeping the closest value per cell.

    Args:
        flat_index: (N,) long tensor of cell indices.
        depth: (N,) depths to scatter.
        num_cells: size of the flat output buffer.
        fill_value: value for cells that receive no point.

    Returns:
        (num_cells,) tensor.
    """
    out = torch.full((num_cells,), float("inf"), device=depth.device, dtype=depth.dtype)
    out.scatter_reduce_(0, flat_index, depth, reduce="amin", include_self=True)
    return torch.where(torch.isinf(out), torch.full_like(out, fill_value), out)


def valid_prompt_mask(prompt: Tensor) -> Tensor:
    """Boolean validity mask of a prompt map (finite and strictly positive)."""
    return torch.isfinite(prompt) & (prompt > 0)


# --------------------------------------------------------------------------- #
# simulated LiDAR on dense (synthetic) depth
# --------------------------------------------------------------------------- #


def sample_velodyne_sensor(height: int, width: int) -> Dict[str, float]:
    """Draws the parameters of one random virtual spinning LiDAR.

    The same sensor should be used for every frame of a clip so that the scan
    pattern is temporally stable, as it is for a real sensor.

    The virtual sensor is a pinhole "LiDAR camera" with a random focal length
    and principal point that is displaced from the RGB camera by a small random
    baseline; its scan lines are evenly spaced in elevation.
    """
    rand = lambda: torch.rand(1).item()  # noqa: E731
    sign = 1.0 if rand() > 0.5 else -1.0
    return {
        "baseline_h": sign * (0.03 + rand() * 0.03),
        "baseline_v": -0.02 + rand() * 0.04,
        "cx": -0.5 * width + rand() * 2.0 * width,
        "cy": 0.5 * height + rand() * 0.2 * height,
        "focal": 1.2 * height + rand() * 0.8 * height,
        "pitch_max": 0.25 + rand() * 0.05,
        "pitch_min": -0.20 - rand() * 0.05,
        "num_lines": int(torch.randint(8, 65, ()).item()),
        "num_points": int(torch.randint(500, 1025, ()).item()),
    }


def simulate_velodyne_prompt(
    depth_hw: Tensor, sensor: Optional[Dict[str, float]] = None, expand_factor: float = 1.5
) -> Tensor:
    """Simulates a spinning-LiDAR prompt from a dense depth map.

    The dense depth is re-projected into a slightly displaced virtual sensor
    (which produces realistic occlusion shadows), sampled along scan lines and
    projected back into the camera.

    Args:
        depth_hw: (H, W) dense depth in metres, 0 where invalid.
        sensor: parameters from :func:`sample_velodyne_sensor`; drawn at random
            when omitted.
        expand_factor: size of the virtual sensor canvas relative to the image.

    Returns:
        (H, W) sparse depth, 0 where there is no return.
    """
    device = depth_hw.device
    height, width = depth_hw.shape
    if sensor is None:
        sensor = sample_velodyne_sensor(height, width)

    h_exp, w_exp = int(expand_factor * height), int(expand_factor * width)

    K_sensor = torch.eye(3, device=device)
    K_sensor[0, 0] = K_sensor[1, 1] = sensor["focal"]
    K_sensor[0, 2] = sensor["cx"]
    K_sensor[1, 2] = sensor["cy"]
    K_canvas = K_sensor.clone()
    K_canvas[0, 2] += (expand_factor - 1.0) / 2.0 * width
    K_canvas[1, 2] += (expand_factor - 1.0) / 2.0 * height
    baseline = torch.tensor([sensor["baseline_h"], sensor["baseline_v"]], device=device)

    v, u = torch.nonzero(valid_prompt_mask(depth_hw), as_tuple=True)
    if v.numel() == 0:
        return torch.zeros_like(depth_hw)
    z = depth_hw[v, u]
    uv1 = torch.stack([u.float(), v.float(), torch.ones_like(z)], dim=0)

    # camera -> displaced virtual sensor canvas (z-buffered)
    points = torch.linalg.inv(K_sensor) @ (uv1 * z)
    points[:2] -= baseline[:, None]
    proj = K_canvas @ points
    pix = torch.round(proj[:2] / (proj[2:3] + 1e-8)).long()
    inside = (pix[0] >= 0) & (pix[0] < w_exp) & (pix[1] >= 0) & (pix[1] < h_exp)
    canvas = _scatter_min_depth(
        pix[1, inside] * w_exp + pix[0, inside], proj[2, inside], h_exp * w_exp
    ).view(h_exp, w_exp)

    # scan pattern of the virtual sensor
    pitch = torch.linspace(sensor["pitch_min"], sensor["pitch_max"], sensor["num_lines"], device=device)
    yaw = torch.linspace(-math.pi / 2.1, math.pi / 2.1, sensor["num_points"], device=device)
    pitch_grid, yaw_grid = torch.meshgrid(pitch, yaw, indexing="xy")
    rays = torch.stack(
        [
            torch.cos(pitch_grid) * torch.sin(yaw_grid),
            torch.sin(pitch_grid),
            torch.cos(pitch_grid) * torch.cos(yaw_grid),
        ],
        dim=0,
    ).view(3, -1)
    ray_proj = K_sensor @ rays
    ray_pix = torch.round(ray_proj[:2] / (ray_proj[2:3] + 1e-8)).long()
    hit = (ray_pix[0] >= 0) & (ray_pix[0] < w_exp) & (ray_pix[1] >= 0) & (ray_pix[1] < h_exp)
    if not hit.any():
        return torch.zeros_like(depth_hw)
    pattern = torch.zeros((h_exp, w_exp), device=device, dtype=torch.bool)
    pattern[ray_pix[1, hit], ray_pix[0, hit]] = True

    # sensor canvas -> camera
    v_s, u_s = torch.nonzero(pattern & (canvas > 0), as_tuple=True)
    if v_s.numel() == 0:
        return torch.zeros_like(depth_hw)
    z_s = canvas[v_s, u_s]
    uv1_s = torch.stack([u_s.float(), v_s.float(), torch.ones_like(z_s)], dim=0)
    points_s = torch.linalg.inv(K_canvas) @ (uv1_s * z_s)
    points_s[:2] += baseline[:, None]
    proj_c = K_sensor @ points_s
    pix_c = torch.round(proj_c[:2] / (proj_c[2:3] + 1e-8)).long()
    inside_c = (pix_c[0] >= 0) & (pix_c[0] < width) & (pix_c[1] >= 0) & (pix_c[1] < height)
    return _scatter_min_depth(
        pix_c[1, inside_c] * width + pix_c[0, inside_c], proj_c[2, inside_c], height * width
    ).view(height, width)


def simulate_scanline_prompt(
    depth_hw: Tensor,
    num_lines: int,
    num_points: int = 200,
    pitch_range: Tuple[float, float] = (-0.5, 0.5),
    yaw_range: Tuple[float, float] = (-math.pi / 2.1, math.pi / 2.1),
) -> Tensor:
    """Deterministic scan-line pattern: keeps dense depth along ``num_lines``.

    Used for validation on synthetic data, where a fixed pattern makes numbers
    comparable between runs. The virtual sensor sits at the camera centre with
    a focal length equal to the image height.

    Args:
        depth_hw: (H, W) dense depth in metres, 0 where invalid.
        num_lines: number of scan lines.
        num_points: number of samples along each line.

    Returns:
        (H, W) sparse depth, 0 where there is no return.
    """
    height, width = depth_hw.shape
    device = depth_hw.device
    pitch = torch.linspace(pitch_range[0], pitch_range[1], num_lines, device=device)
    yaw = torch.linspace(yaw_range[0], yaw_range[1], num_points, device=device)
    pitch_grid, yaw_grid = torch.meshgrid(pitch, yaw, indexing="ij")
    x = torch.cos(pitch_grid) * torch.sin(yaw_grid)
    y = torch.sin(pitch_grid)
    z = torch.cos(pitch_grid) * torch.cos(yaw_grid)
    u = torch.round(height * x / (z + 1e-8) + 0.5 * width).long().flatten()
    v = torch.round(height * y / (z + 1e-8) + 0.5 * height).long().flatten()
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    pattern = torch.zeros_like(depth_hw, dtype=torch.bool)
    pattern[v[inside], u[inside]] = True
    return torch.where(pattern & valid_prompt_mask(depth_hw), depth_hw, torch.zeros_like(depth_hw))


# --------------------------------------------------------------------------- #
# real LiDAR projections
# --------------------------------------------------------------------------- #


def sample_lidar_lines(
    depth_1hw: Tensor,
    K_b44: Tensor,
    num_bins: int = 64,
    keep_every: int = 2,
    pitch_range: Optional[Tuple[float, float]] = None,
) -> Tensor:
    """Thins a projected LiDAR sweep to fewer scan lines.

    Every return is back-projected with the camera intrinsics and assigned to
    one of ``num_bins`` equally spaced elevation bins; only every
    ``keep_every``-th bin is kept. With ``num_bins`` equal to the number of
    beams of the sensor this approximates a sensor with
    ``num_bins / keep_every`` beams.

    Args:
        depth_1hw: (1, H, W) projected LiDAR depth; 0 or NaN where empty.
        K_b44: (4, 4) intrinsics of the image the depth is given in.
        num_bins: number of elevation bins.
        keep_every: keep bins whose index is a multiple of this value.
        pitch_range: optional fixed (min, max) elevation in radians used for
            binning. When omitted the range of the current sweep is used, which
            is what the original experiments did but lets the kept lines drift
            slightly from frame to frame.

    Returns:
        (1, H, W) sparse depth with 0 where there is no kept return.
    """
    assert depth_1hw.ndim == 3 and depth_1hw.shape[0] == 1
    depth = depth_1hw[0]
    out = torch.zeros_like(depth_1hw)

    v, u = torch.nonzero(valid_prompt_mask(depth), as_tuple=True)
    if v.numel() == 0:
        return out
    z = depth[v, u]

    K_inv = torch.inverse(K_b44[:3, :3].to(z.dtype))
    uv1 = torch.stack([u.to(z.dtype), v.to(z.dtype), torch.ones_like(z)], dim=0)
    points = (K_inv @ uv1) * z.unsqueeze(0)
    pitch = torch.asin(torch.clamp(points[1] / points.norm(dim=0).clamp_min(1e-8), -1.0, 1.0))

    if pitch_range is None:
        pitch_min, pitch_max = pitch.min(), pitch.max()
    else:
        pitch_min = torch.as_tensor(pitch_range[0], dtype=z.dtype, device=z.device)
        pitch_max = torch.as_tensor(pitch_range[1], dtype=z.dtype, device=z.device)
    if pitch_max <= pitch_min:
        out[0, v[0], u[0]] = z[0]
        return out

    bins = torch.floor((pitch - pitch_min) / ((pitch_max - pitch_min) / num_bins)).long()
    in_range = (bins >= 0) & (bins <= num_bins)
    bins = bins.clamp(0, num_bins - 1)
    keep = in_range & (bins % max(1, int(keep_every)) == 0)
    out[0, v[keep], u[keep]] = z[keep]
    return out


def resize_sparse_depth(prompt_1hw: Tensor, target_hw: Tuple[int, int]) -> Tensor:
    """Resizes a sparse depth map by moving every valid sample.

    Interpolation would blend measurements with empty pixels, so each valid
    sample is moved to its nearest target pixel instead; when several samples
    land on one pixel the closest depth wins.

    Args:
        prompt_1hw: (1, H, W) sparse depth, 0 where empty.
        target_hw: (H_new, W_new).

    Returns:
        (1, H_new, W_new) sparse depth.
    """
    assert prompt_1hw.ndim == 3 and prompt_1hw.shape[0] == 1
    height, width = prompt_1hw.shape[-2:]
    new_h, new_w = target_hw
    if (height, width) == (new_h, new_w):
        return prompt_1hw.clone()

    v, u = torch.nonzero(valid_prompt_mask(prompt_1hw[0]), as_tuple=True)
    if v.numel() == 0:
        return prompt_1hw.new_zeros((1, new_h, new_w))
    z = prompt_1hw[0, v, u]
    v_new = torch.clamp(torch.round(v.float() * (new_h / height)).long(), 0, new_h - 1)
    u_new = torch.clamp(torch.round(u.float() * (new_w / width)).long(), 0, new_w - 1)
    return _scatter_min_depth(v_new * new_w + u_new, z, new_h * new_w).view(1, new_h, new_w)


def downsample_sparse_depth(
    depth: Tensor, mask: Tensor, rate: int, method: str = "min"
) -> Tuple[Tensor, Tensor]:
    """Downsamples a sparse depth map without mixing in empty pixels.

    Args:
        depth: (..., 1, H, W) sparse depth.
        mask: (..., 1, H, W) float validity mask.
        rate: integer downsampling factor.
        method: ``"min"`` keeps the closest valid depth per cell, ``"mean"``
            averages the valid depths.

    Returns:
        depth and float mask at (..., 1, H / rate, W / rate).
    """
    lead = depth.shape[:-3]
    depth = depth.reshape(-1, *depth.shape[-3:])
    mask = mask.reshape(-1, *mask.shape[-3:]).to(depth.dtype)

    if method == "mean":
        mask_sum = F.avg_pool2d(mask, rate) * rate**2
        depth_sum = F.avg_pool2d(depth * mask, rate) * rate**2
        depth_down = torch.where(mask_sum > 0, depth_sum / mask_sum.clamp_min(1.0), torch.zeros_like(depth_sum))
        mask_down = (mask_sum > 0).to(depth.dtype)
    elif method == "min":
        depth_inf = torch.where(mask > 0, depth, torch.full_like(depth, float("inf")))
        depth_min = -F.max_pool2d(-depth_inf, rate, stride=rate)
        mask_down = F.max_pool2d(mask, rate, stride=rate)
        depth_down = torch.where(mask_down > 0, depth_min, torch.zeros_like(depth_min))
    else:
        raise ValueError(f"Unknown sparse downsampling method {method}")

    return (
        depth_down.reshape(*lead, *depth_down.shape[-3:]),
        mask_down.reshape(*lead, *mask_down.shape[-3:]),
    )


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #


def augment_sparse_prompt(
    prompt: Tensor,
    noise_sigma: float = 0.0,
    drop_range: Tuple[float, float] = (0.0, 0.0),
    outlier_ratio: float = 0.0,
) -> Tensor:
    """Corrupts a prompt the way a real sensor would.

    Args:
        prompt: (..., H, W) sparse depth, 0 where empty.
        noise_sigma: std of multiplicative Gaussian range noise
            (``depth * (1 + sigma * n)``).
        drop_range: a fraction drawn uniformly from this range of the valid
            samples is removed.
        outlier_ratio: fraction of the remaining samples replaced by a depth
            drawn log-uniformly between half and twice the true depth
            (spurious returns, mixed pixels at depth edges).

    Returns:
        augmented prompt of the same shape.
    """
    prompt = prompt.clone()
    valid = valid_prompt_mask(prompt)
    if not valid.any():
        return prompt

    if drop_range[1] > 0:
        ratio = torch.empty(1).uniform_(drop_range[0], drop_range[1]).item()
        valid = valid & (torch.rand_like(prompt) >= ratio)

    if noise_sigma > 0:
        prompt = prompt * (1.0 + noise_sigma * torch.randn_like(prompt))

    if outlier_ratio > 0:
        outlier = valid & (torch.rand_like(prompt) < outlier_ratio)
        factor = torch.exp((torch.rand_like(prompt) * 2.0 - 1.0) * math.log(2.0))
        prompt = torch.where(outlier, prompt * factor, prompt)

    return torch.where(valid, prompt.clamp_min(1e-3), torch.zeros_like(prompt))


def limit_prompt_field_of_view(prompt: Tensor, top_fraction: float = 0.0, bottom_fraction: float = 0.0) -> Tensor:
    """Removes prompt samples in the top / bottom part of the image.

    A roof-mounted automotive LiDAR sees little above the horizon and nothing
    right in front of the bumper; this emulates both blind zones.

    Args:
        prompt: (..., H, W) sparse depth.
        top_fraction: fraction of rows cleared at the top.
        bottom_fraction: fraction of rows cleared at the bottom.
    """
    prompt = prompt.clone()
    height = prompt.shape[-2]
    top = int(round(height * top_fraction))
    bottom = int(round(height * bottom_fraction))
    if top > 0:
        prompt[..., :top, :] = 0
    if bottom > 0:
        prompt[..., height - bottom :, :] = 0
    return prompt


# --------------------------------------------------------------------------- #
# geometry: moving prompts between views
# --------------------------------------------------------------------------- #


def transfer_sparse_depth(
    src_depth_bk1hw: Tensor,
    src_invK_bk44: Tensor,
    cur_K_b44: Tensor,
    cur_cam_T_src_cam_bk44: Tensor,
    min_depth: float = 1e-2,
) -> Tuple[Tensor, Tensor]:
    """Moves sparse depth measured in source views into the reference view.

    Every valid source sample is lifted to 3D with the source intrinsics,
    transformed with the known relative pose and splatted into the reference
    image with a z-buffer. For a static scene the result is an exact prompt for
    the reference view even if that view has no measurement of its own.

    Samples of moving objects end up at the place the object *was* and sparse
    points cannot occlude each other properly, so transferred prompts are less
    reliable than direct ones; the network treats them as a separate input.

    Args:
        src_depth_bk1hw: (B, K, 1, H, W) sparse depth of K source views.
        src_invK_bk44: (B, K, 4, 4) inverse intrinsics of the source views at
            the resolution of ``src_depth_bk1hw``.
        cur_K_b44: (B, 4, 4) intrinsics of the reference view at that resolution.
        cur_cam_T_src_cam_bk44: (B, K, 4, 4) source-camera to reference-camera
            transforms.
        min_depth: transferred points closer than this are dropped.

    Returns:
        per_view_bk1hw: (B, K, 1, H, W) what each source view contributes.
        merged_b1hw: (B, 1, H, W) closest transferred depth over all views.
    """
    batch, num_views, _, height, width = src_depth_bk1hw.shape
    device = src_depth_bk1hw.device
    dtype = torch.float32

    depth = src_depth_bk1hw.reshape(batch * num_views, height * width).to(dtype)
    invK = src_invK_bk44.reshape(batch * num_views, 4, 4).to(dtype)
    K = cur_K_b44.to(dtype).unsqueeze(1).expand(-1, num_views, -1, -1).reshape(-1, 4, 4)
    T = cur_cam_T_src_cam_bk44.reshape(batch * num_views, 4, 4).to(dtype)

    view_idx, pix_idx = torch.nonzero(valid_prompt_mask(depth), as_tuple=True)
    per_view = torch.zeros((batch * num_views * height * width,), device=device, dtype=dtype)
    if view_idx.numel() > 0:
        z = depth[view_idx, pix_idx]
        # pixel centres sit at +0.5, matching BackprojectDepth.
        u = (pix_idx % width).to(dtype) + 0.5
        v = torch.div(pix_idx, width, rounding_mode="floor").to(dtype) + 0.5
        pix = torch.stack([u, v, torch.ones_like(u)], dim=1)

        points_src = torch.einsum("nij,nj->ni", invK[view_idx, :3, :3], pix) * z[:, None]
        points_cur = torch.einsum("nij,nj->ni", T[view_idx, :3, :3], points_src) + T[view_idx, :3, 3]
        proj = torch.einsum("nij,nj->ni", K[view_idx, :3, :3], points_cur)

        z_cur = proj[:, 2]
        in_front = z_cur > min_depth
        z_safe = torch.where(in_front, z_cur, torch.ones_like(z_cur))
        u_cur = torch.floor(proj[:, 0] / z_safe).long()
        v_cur = torch.floor(proj[:, 1] / z_safe).long()
        inside = in_front & (u_cur >= 0) & (u_cur < width) & (v_cur >= 0) & (v_cur < height)

        flat = view_idx[inside] * (height * width) + v_cur[inside] * width + u_cur[inside]
        per_view = _scatter_min_depth(flat, z_cur[inside], per_view.numel())

    per_view_bk1hw = per_view.view(batch, num_views, 1, height, width)
    merged = torch.where(per_view_bk1hw > 0, per_view_bk1hw, torch.full_like(per_view_bk1hw, float("inf")))
    merged = merged.min(dim=1).values
    merged_b1hw = torch.where(torch.isinf(merged), torch.zeros_like(merged), merged)
    return per_view_bk1hw.to(src_depth_bk1hw.dtype), merged_b1hw.to(src_depth_bk1hw.dtype)


# --------------------------------------------------------------------------- #
# prompt dropout (training) and prompt switches (evaluation)
# --------------------------------------------------------------------------- #

PROMPT_MODES = ("both", "cur_only", "src_only", "none")


def sample_prompt_mode(probabilities: Tuple[float, float, float, float]) -> str:
    """Draws one of :data:`PROMPT_MODES` with the given probabilities."""
    probs = torch.tensor(probabilities, dtype=torch.float32)
    assert probs.numel() == len(PROMPT_MODES) and probs.sum() > 0
    return PROMPT_MODES[int(torch.multinomial(probs / probs.sum(), 1).item())]


def apply_prompt_mode(cur_data: dict, src_data: dict, mode: str) -> None:
    """Zeroes prompts in place according to ``mode``.

    Args:
        cur_data, src_data: batch dictionaries holding ``sparse_prompt_b1hw``
            and ``sparse_prompt_mask_b1hw``.
        mode: ``"both"`` keeps everything, ``"cur_only"`` removes the prompts
            of the source views, ``"src_only"`` removes the prompt of the
            reference view, ``"none"`` removes all prompts.
    """
    assert mode in PROMPT_MODES, f"Unknown prompt mode {mode}"
    drop = []
    if mode in ("src_only", "none"):
        drop.append(cur_data)
    if mode in ("cur_only", "none"):
        drop.append(src_data)
    for data in drop:
        for key in ("sparse_prompt_b1hw", "sparse_prompt_mask_b1hw"):
            if key in data:
                data[key] = torch.zeros_like(data[key])


# --------------------------------------------------------------------------- #
# encodings shared by the network
# --------------------------------------------------------------------------- #


def normalized_log_prompt(
    prompt: Tensor, mask: Tensor, min_depth: Tensor, max_depth: Tensor, eps: float = 1e-6
) -> Tensor:
    """Encodes a prompt in the space the network predicts in.

    The network outputs a logit ``x`` that is turned into depth with
    ``exp(log(min) + log(max / min) * sigmoid(x))``. Encoding the prompt with
    the inverse mapping makes "copy the prompt" a trivial function.

    Args:
        prompt: (..., 1, H, W) sparse depth in metres.
        mask: (..., 1, H, W) float validity mask.
        min_depth, max_depth: broadcastable depth range of the frame.

    Returns:
        (..., 2, H, W): the logit (0 where invalid) and the mask.
    """
    depth = torch.minimum(torch.maximum(prompt, min_depth), max_depth)
    position = (torch.log(depth) - torch.log(min_depth)) / (torch.log(max_depth) - torch.log(min_depth))
    logit = torch.logit(position.clamp(eps, 1.0 - eps))
    logit = torch.where(mask > 0.5, logit, torch.zeros_like(logit))
    return torch.cat([logit, mask.to(logit.dtype)], dim=-3)


def prompt_depth_range(
    prompts: Tensor, near_scale: float = 0.5, far_scale: float = 2.0, quantile: float = 0.0
) -> Tuple[Tensor, Tensor, Tensor]:
    """Depth range bracketing the prompts of each batch element.

    Args:
        prompts: (B, ...) sparse depth; every trailing dimension is pooled.
        near_scale, far_scale: margins applied to the closest / farthest prompt.
        quantile: when > 0 the ``quantile`` and ``1 - quantile`` quantiles are
            used instead of min and max, which resists outliers.

    Returns:
        min_depth (B,), max_depth (B,), has_prompt (B,) bool. Entries without
        any prompt hold NaN in the first two tensors.
    """
    flat = prompts.flatten(1).float()
    valid = valid_prompt_mask(flat)
    has_prompt = valid.any(dim=1)
    if quantile > 0:
        nan_flat = torch.where(valid, flat, torch.full_like(flat, float("nan")))
        near = torch.nanquantile(nan_flat, quantile, dim=1)
        far = torch.nanquantile(nan_flat, 1.0 - quantile, dim=1)
    else:
        near = torch.where(valid, flat, torch.full_like(flat, float("inf"))).amin(dim=1)
        far = torch.where(valid, flat, torch.full_like(flat, float("-inf"))).amax(dim=1)
    nan = torch.full_like(near, float("nan"))
    return (
        torch.where(has_prompt, near * near_scale, nan),
        torch.where(has_prompt, far * far_scale, nan),
        has_prompt,
    )
