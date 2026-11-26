"""Temporal consistency metric for depth sequences with known camera motion.

DriveMVS addition; not part of the upstream MVSAnywhere code base.
"""

import torch
from torch import Tensor


def reproject_depth(depth_hw: Tensor, K_33: Tensor, tgt_cam_T_src_cam_44: Tensor) -> Tensor:
    """Forward-warps a depth map into another camera.

    Every pixel is lifted to 3D, moved with the relative pose and splatted to
    its nearest pixel in the target view; where several points land on one
    pixel the closest wins.

    Args:
        depth_hw: (H, W) depth of the source frame, <= 0 or non-finite where
            invalid.
        K_33: intrinsics shared by both frames.
        tgt_cam_T_src_cam_44: transform from source to target camera.

    Returns:
        (H, W) depth as seen from the target camera, 0 where nothing landed.
    """
    height, width = depth_hw.shape
    device, dtype = depth_hw.device, torch.float64
    depth = depth_hw.to(dtype)
    K = K_33.to(dtype)
    T = tgt_cam_T_src_cam_44.to(dtype)

    valid = torch.isfinite(depth) & (depth > 0)
    v, u = torch.nonzero(valid, as_tuple=True)
    z = depth[v, u]
    pix = torch.stack([u.to(dtype) + 0.5, v.to(dtype) + 0.5, torch.ones_like(z)], dim=0)
    points = (torch.linalg.inv(K) @ pix) * z
    points = T[:3, :3] @ points + T[:3, 3:4]
    proj = K @ points

    z_t = proj[2]
    front = z_t > 1e-6
    u_t = torch.floor(proj[0] / z_t.clamp_min(1e-6)).long()
    v_t = torch.floor(proj[1] / z_t.clamp_min(1e-6)).long()
    inside = front & (u_t >= 0) & (u_t < width) & (v_t >= 0) & (v_t < height)

    out = torch.full((height * width,), float("inf"), device=device, dtype=dtype)
    out.scatter_reduce_(0, v_t[inside] * width + u_t[inside], z_t[inside], reduce="amin")
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out.view(height, width).to(depth_hw.dtype)


def lookup_depth(depth_hw: Tensor, K_33: Tensor, other_cam_T_cam_44: Tensor, other_depth_hw: Tensor):
    """For every pixel of a frame, finds the depth the other frame assigns to
    the same 3D point.

    The pixel is lifted with its own depth, moved into the other camera and
    projected; the other depth map is read at that pixel.

    Returns:
        expected: (H, W) depth of the pixel's 3D point in the other camera.
        observed: (H, W) depth the other frame holds at the projection.
        valid: (H, W) bool, the point projects inside the other image and
            both depths are valid.
    """
    height, width = depth_hw.shape
    dtype = torch.float64
    depth = depth_hw.to(dtype)
    K = K_33.to(dtype)
    T = other_cam_T_cam_44.to(dtype)

    vv, uu = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=dtype),
        torch.arange(width, device=depth.device, dtype=dtype),
        indexing="ij",
    )
    pix = torch.stack([uu + 0.5, vv + 0.5, torch.ones_like(uu)], dim=0).reshape(3, -1)
    safe_depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, torch.ones_like(depth))
    points = (torch.linalg.inv(K) @ pix) * safe_depth.reshape(1, -1)
    points = T[:3, :3] @ points + T[:3, 3:4]
    proj = K @ points

    expected = proj[2]
    front = expected > 1e-6
    # continuous pixel coordinates of the projection (pixel centres at +0.5)
    x = proj[0] / expected.clamp_min(1e-6) - 0.5
    y = proj[1] / expected.clamp_min(1e-6) - 0.5
    inside = front & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)

    # Read the other depth map with bilinear interpolation of inverse depth
    # (exact on planes). Nearest-pixel lookup is not good enough: on the road
    # far ahead one pixel row spans metres of depth.
    other = other_depth_hw.to(dtype)
    other_valid = torch.isfinite(other) & (other > 0)
    inv_other = torch.where(other_valid, 1.0 / other.clamp_min(1e-6), torch.zeros_like(other))
    grid = torch.stack([2 * (x + 0.5) / width - 1, 2 * (y + 0.5) / height - 1], dim=-1).view(1, 1, -1, 2)
    sample = lambda t: torch.nn.functional.grid_sample(  # noqa: E731
        t.view(1, 1, height, width), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).view(-1)
    inv_observed = sample(inv_other)
    # only where all four neighbours are valid
    full_support = sample(other_valid.to(dtype)) > 0.999
    observed = 1.0 / inv_observed.clamp_min(1e-9)

    valid = inside & full_support & (torch.isfinite(depth) & (depth > 0)).reshape(-1)
    return expected.view(height, width), observed.view(height, width), valid.view(height, width)


def temporal_alignment_error(
    depth_a_hw: Tensor,
    depth_b_hw: Tensor,
    K_33: Tensor,
    b_cam_T_a_cam_44: Tensor,
    mask_a_hw: Tensor = None,
    mask_b_hw: Tensor = None,
    occlusion_tolerance: float = None,
):
    """Temporal alignment error (TAE) between two frames.

    Every pixel of frame a is lifted with its predicted depth and moved into
    camera b with the known relative pose. The depth this 3D point has in
    camera b is compared with the depth predicted for frame b at the pixel it
    projects to (absolute relative difference). The same is done from b to a
    and the two directions are averaged. A temporally consistent, metric
    sequence of a static scene scores 0 up to occlusions.

    This follows the idea of the TAE used in "Depth Any Video" and the
    DriveMVS paper but reads the other depth map at the projected pixel
    instead of forward splatting, which avoids the holes splatting leaves
    when the camera moves forward. Note for anyone comparing with the first
    arXiv version of DriveMVS: the script behind those numbers treated
    camera-to-world poses as world-to-camera and counted pixels without a
    reprojected point as depth 0, so its values are not comparable.

    Args:
        depth_a_hw, depth_b_hw: (H, W) predicted depths.
        K_33: shared intrinsics.
        b_cam_T_a_cam_44: transform from camera a to camera b.
        mask_a_hw, mask_b_hw: optional bool masks restricting the pixels of
            the respective frame that are lifted.
        occlusion_tolerance: if given, pixel pairs whose relative difference
            exceeds this value are regarded as occlusions / moving objects
            and left out.

    Returns:
        scalar tensor, or None if no pixel could be compared.
    """

    def one_way(src, tgt, T, src_mask):
        expected, observed, valid = lookup_depth(src, K_33, T, tgt)
        if src_mask is not None:
            valid = valid & src_mask
        if not valid.any():
            return None
        error = (expected[valid] - observed[valid]).abs() / observed[valid]
        if occlusion_tolerance is not None:
            error = error[error < occlusion_tolerance]
            if error.numel() == 0:
                return None
        return error.mean()

    T_ba = b_cam_T_a_cam_44.double()
    forward = one_way(depth_a_hw, depth_b_hw, T_ba, mask_a_hw)
    backward = one_way(depth_b_hw, depth_a_hw, torch.linalg.inv(T_ba), mask_b_hw)
    if forward is None or backward is None:
        return None
    return 0.5 * (forward + backward)
