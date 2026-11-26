"""Unit tests for the DriveMVS prompt utilities and the temporal metric.

Run with:  pytest tests
"""

import math

import torch

from mvsanywhere.utils import prompt_utils as pu
from mvsanywhere.utils.temporal_metrics import temporal_alignment_error

H, W = 96, 128


def intrinsics():
    K = torch.eye(4)
    K[0, 0] = K[1, 1] = 100.0
    K[0, 2], K[1, 2] = W / 2, H / 2
    return K


def plane_depth(K, cam_T_world, normal, offset):
    """Depth map of the world plane n.X = offset seen by a camera."""
    world_T_cam = torch.linalg.inv(cam_T_world)
    vv, uu = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
    rays_cam = torch.linalg.inv(K[:3, :3]) @ torch.stack([uu, vv, torch.ones_like(uu)]).reshape(3, -1)
    rays_world = world_T_cam[:3, :3] @ rays_cam
    origin = world_T_cam[:3, 3]
    n = torch.tensor(normal, dtype=torch.float32)
    t = (offset - n @ origin) / (n @ rays_world)  # rays have z = 1 in the camera frame
    return t.reshape(H, W)


def two_views():
    K = intrinsics()
    cam_a = torch.eye(4)
    cam_b = torch.eye(4)
    cam_b[:3, 3] = torch.tensor([-0.4, 0.05, -0.8])  # world -> camera b
    normal, offset = (0.1, -0.3, 1.0), 12.0  # slanted wall in front of the cameras
    return K, cam_a, cam_b, plane_depth(K, cam_a, normal, offset), plane_depth(K, cam_b, normal, offset)


def test_transfer_moves_prompts_exactly_for_a_static_scene():
    K, cam_a, cam_b, depth_a, depth_b = two_views()
    sparse_a = torch.zeros_like(depth_a)
    sparse_a[::6, ::5] = depth_a[::6, ::5]
    per_view, merged = pu.transfer_sparse_depth(
        sparse_a[None, None, None],
        torch.linalg.inv(K)[None, None],
        K[None],
        (cam_b @ torch.linalg.inv(cam_a))[None, None],
    )
    assert per_view.shape == (1, 1, 1, H, W) and merged.shape == (1, 1, H, W)
    valid = merged[0, 0] > 0
    assert valid.sum() > 100
    # the transferred value is the depth of the 3D point in view b; view b's own
    # depth at the (rounded) pixel differs by at most the slope over one pixel.
    rel = (merged[0, 0][valid] - depth_b[valid]).abs() / depth_b[valid]
    assert rel.max() < 0.02


def test_transfer_keeps_closest_point_and_ignores_empty_views():
    K = intrinsics()
    src = torch.zeros(1, 2, 1, H, W)
    src[0, 0, 0, 40, 60] = 10.0
    src[0, 0, 0, 40, 61] = 4.0  # nearly the same pixel after rounding, closer
    eye = torch.eye(4)[None, None].expand(1, 2, 4, 4)
    per_view, merged = pu.transfer_sparse_depth(src, torch.linalg.inv(K)[None, None].expand(1, 2, 4, 4), K[None], eye)
    assert per_view[0, 1].abs().sum() == 0
    assert merged[0, 0, 40, 60] == 10.0 and merged[0, 0, 40, 61] == 4.0


def test_prompt_modes():
    cur = {"sparse_prompt_b1hw": torch.ones(2, 1, 4, 4), "sparse_prompt_mask_b1hw": torch.ones(2, 1, 4, 4)}
    src = {"sparse_prompt_b1hw": torch.ones(2, 3, 1, 4, 4), "sparse_prompt_mask_b1hw": torch.ones(2, 3, 1, 4, 4)}
    expected = {"both": (1, 1), "cur_only": (1, 0), "src_only": (0, 1), "none": (0, 0)}
    for mode, (keep_cur, keep_src) in expected.items():
        c = {k: v.clone() for k, v in cur.items()}
        s = {k: v.clone() for k, v in src.items()}
        pu.apply_prompt_mode(c, s, mode)
        assert c["sparse_prompt_b1hw"].max() == keep_cur and s["sparse_prompt_b1hw"].max() == keep_src
        assert c["sparse_prompt_mask_b1hw"].max() == keep_cur and s["sparse_prompt_mask_b1hw"].max() == keep_src
    assert pu.sample_prompt_mode((0.0, 0.0, 1.0, 0.0)) == "src_only"


def test_simulated_lidar_only_returns_true_depths():
    torch.manual_seed(0)
    depth = torch.rand(H, W) * 40 + 2
    depth[:10] = 0  # invalid (sky)
    for prompt in (pu.simulate_velodyne_prompt(depth), pu.simulate_scanline_prompt(depth, 16)):
        valid = prompt > 0
        assert 0 < valid.float().mean() < 0.5
        assert not valid[:10].any()
    scan = pu.simulate_scanline_prompt(depth, 16)
    assert torch.equal(scan[scan > 0], depth[scan > 0])


def test_sparse_resize_and_downsample_do_not_blend():
    prompt = torch.zeros(1, 40, 80)
    prompt[0, 10, 20], prompt[0, 30, 70] = 5.0, 9.0
    resized = pu.resize_sparse_depth(prompt, (20, 40))
    assert sorted(resized[resized > 0].tolist()) == [5.0, 9.0]
    down, mask = pu.downsample_sparse_depth(prompt[None], (prompt[None] > 0).float(), 4)
    assert sorted(down[down > 0].tolist()) == [5.0, 9.0] and mask.sum() == 2


def test_normalized_log_prompt_inverts_the_depth_decoding():
    lo, hi = torch.tensor(1.0), torch.tensor(100.0)
    depth = torch.tensor([[[[2.0, 50.0], [0.0, 10.0]]]])
    mask = (depth > 0).float()
    encoded = pu.normalized_log_prompt(depth, mask, lo, hi)
    decoded = torch.exp(torch.log(lo) + torch.log(hi / lo) * torch.sigmoid(encoded[:, :1]))
    assert torch.allclose(decoded[mask > 0], depth[mask > 0], rtol=1e-4)
    assert encoded[0, 0, 1, 0] == 0 and torch.equal(encoded[:, 1:], mask)


def test_prompt_depth_range():
    prompts = torch.zeros(2, 1, 8, 8)
    prompts[0, 0, 0, :4] = torch.tensor([4.0, 8.0, 20.0, 40.0])
    lo, hi, has = pu.prompt_depth_range(prompts, 0.5, 2.0)
    assert has.tolist() == [True, False]
    assert math.isclose(lo[0].item(), 2.0) and math.isclose(hi[0].item(), 80.0)


def test_temporal_alignment_error():
    K, cam_a, cam_b, depth_a, depth_b = two_views()
    T = cam_b @ torch.linalg.inv(cam_a)
    consistent = temporal_alignment_error(depth_a, depth_b, K[:3, :3], T)
    flicker = temporal_alignment_error(depth_a * 1.05, depth_b * 0.95, K[:3, :3], T)
    wrong_pose = temporal_alignment_error(depth_a, depth_b, K[:3, :3], torch.linalg.inv(T))
    assert consistent < 1e-3
    assert 0.08 < flicker < 0.12
    assert wrong_pose > 10 * consistent
