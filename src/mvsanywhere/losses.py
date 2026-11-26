import kornia
import torch
import torch.jit as jit
import torch.nn.functional as F
from torch import Tensor, nn

from mvsanywhere.utils.generic_utils import pyrdown
from mvsanywhere.utils.geometry_utils import BackprojectDepth, Project3D


class MSGradientLoss(nn.Module):
    def __init__(self, num_scales: int = 4):
        super().__init__()

        self.num_scales = num_scales

    def forward(self, depth_gt: Tensor, depth_pred: Tensor) -> Tensor:
        # Create the gradient pyramids
        depth_pred_pyr = pyrdown(depth_pred, self.num_scales)
        depth_gtn_pyr = pyrdown(depth_gt, self.num_scales)

        grad_loss = torch.tensor(0, dtype=depth_gt.dtype, device=depth_gt.device)
        for depth_pred_down, depth_gtn_down in zip(depth_pred_pyr, depth_gtn_pyr):
            depth_gtn_grad = kornia.filters.spatial_gradient(1.0 / (depth_gtn_down + 1e-6))

            mask_down_b = depth_gtn_grad.isfinite().all(dim=1, keepdim=True)

            depth_pred_grad = kornia.filters.spatial_gradient(1.0 / (depth_pred_down + 1e-6)).masked_select(
                mask_down_b
            )

            grad_error = torch.abs(depth_pred_grad - depth_gtn_grad.masked_select(mask_down_b))
            grad_loss += torch.mean(grad_error)


        return grad_loss


class ScaleInvariantLoss(jit.ScriptModule):
    def __init__(self, si_lambda: float = 0.85):
        super().__init__()

        self.si_lambda = si_lambda

    @jit.script_method
    def forward(self, log_depth_gt: Tensor, log_depth_pred: Tensor) -> Tensor:
        # Scale invariant loss from Eigen, implementation is from AdaBins
        log_diff = log_depth_gt - log_depth_pred
        si_loss = torch.sqrt((log_diff**2).mean() - self.si_lambda * (log_diff.mean() ** 2))

        return si_loss


class NormalsLoss(nn.Module):
    def forward(self, normals_gt_b3hw: Tensor, normals_pred_b3hw: Tensor) -> Tensor:
        normals_mask_b1hw = torch.logical_and(
            normals_gt_b3hw.isfinite().all(dim=1, keepdim=True),
            normals_pred_b3hw.isfinite().all(dim=1, keepdim=True),
        )

        normals_pred_b3hw = normals_pred_b3hw.masked_fill(~normals_mask_b1hw, 1.0)
        normals_gt_b3hw = normals_gt_b3hw.masked_fill(~normals_mask_b1hw, 1.0)

        with torch.cuda.amp.autocast(enabled=False):
            normals_dot_b1hw = 0.5 * (
                1.0
                - torch.einsum(
                    "bchw, bchw -> bhw",
                    normals_pred_b3hw,
                    normals_gt_b3hw,
                )
            ).unsqueeze(1)
        normals_loss = normals_dot_b1hw.masked_select(normals_mask_b1hw).mean()

        return normals_loss


class MVDepthLoss(nn.Module):
    def __init__(self, height, width):
        super().__init__()

        self.height = height
        self.width = width

        self.backproject = BackprojectDepth(self.height, self.width)
        self.project = Project3D()

    def _check_warped_image(
        self,
        image_name,
        cur_depth_b1hw,
        cur_img_b3hw,
        src_img_b3hw,
        cur_invK_b44,
        src_K_b44,
        cur_world_T_cam_b44,
        src_cam_T_world_b44,
    ):
        import matplotlib.pyplot as plt
        depth_height, depth_width = cur_depth_b1hw.shape[2:]

        cur_cam_points_b4N = self.backproject(cur_depth_b1hw, cur_invK_b44)
        world_points_b4N = cur_world_T_cam_b44 @ cur_cam_points_b4N

        # Compute valid mask
        src_cam_points_b3N = self.project(world_points_b4N, src_K_b44, src_cam_T_world_b44)
        # src_cam_points_b3N = self.project(src_cam_T_world_b44 @ world_points_b4N, src_K_b44, torch.stack([torch.eye(4) for i in range(4)]).cuda())

        cam_points_b3hw = src_cam_points_b3N.view(-1, 3, depth_height, depth_width)
        pix_coords_b2hw = cam_points_b3hw[:, :2]

        uv_coords = pix_coords_b2hw.permute(0, 2, 3, 1) / torch.tensor(
            [depth_width, depth_height]
        ).view(1, 1, 1, 2).type_as(pix_coords_b2hw)
        uv_coords = 2 * uv_coords - 1

        src_img_sampled_b1hw = F.grid_sample(
            input=src_img_b3hw,
            grid=uv_coords,
            padding_mode="zeros",
            mode="nearest",
            align_corners=False,
        )
        src_img_sampled_b1hw = F.upsample(
            src_img_sampled_b1hw, scale_factor=2.0
        )
        for i in range(cur_img_b3hw.shape[0]):
            plt.imsave(f"{image_name}_{i}_cur.png", cur_img_b3hw[i, 0].detach().cpu().numpy(), cmap='gray')
            plt.imsave(f"{image_name}_{i}_src.png", src_img_b3hw[i, 0].detach().cpu().numpy(), cmap='gray')
            plt.imsave(f"{image_name}_{i}_smp.png", src_img_sampled_b1hw[i, 0].detach().cpu().numpy(), cmap='gray')



    def get_valid_mask(
        self,
        cur_depth_b1hw,
        src_depth_b1hw,
        cur_invK_b44,
        src_K_b44,
        cur_world_T_cam_b44,
        src_cam_T_world_b44,
    ):
        depth_height, depth_width = cur_depth_b1hw.shape[2:]

        cur_cam_points_b4N = self.backproject(cur_depth_b1hw, cur_invK_b44)
        world_points_b4N = cur_world_T_cam_b44 @ cur_cam_points_b4N

        # Compute valid mask
        src_cam_points_b3N = self.project(world_points_b4N, src_K_b44, src_cam_T_world_b44)

        cam_points_b3hw = src_cam_points_b3N.view(-1, 3, depth_height, depth_width)
        pix_coords_b2hw = cam_points_b3hw[:, :2]
        proj_src_depths_b1hw = cam_points_b3hw[:, 2:]

        uv_coords = pix_coords_b2hw.permute(0, 2, 3, 1) / torch.tensor(
            [depth_width, depth_height]
        ).view(1, 1, 1, 2).type_as(pix_coords_b2hw)
        uv_coords = 2 * uv_coords - 1

        src_depth_sampled_b1hw = F.grid_sample(
            input=src_depth_b1hw,
            grid=uv_coords,
            padding_mode="zeros",
            mode="nearest",
            align_corners=False,
        )

        valid_mask_b1hw = proj_src_depths_b1hw < 1.05 * src_depth_sampled_b1hw
        valid_mask_b1hw = torch.logical_and(valid_mask_b1hw, proj_src_depths_b1hw > 0)
        valid_mask_b1hw = torch.logical_and(valid_mask_b1hw, src_depth_sampled_b1hw > 0)

        return valid_mask_b1hw, src_depth_sampled_b1hw

    def get_error_for_pair(
        self,
        depth_pred_b1hw,
        cur_depth_b1hw,
        src_depth_b1hw,
        cur_invK_b44,
        src_K_b44,
        cur_world_T_cam_b44,
        src_cam_T_world_b44,
    ):
        depth_height, depth_width = cur_depth_b1hw.shape[2:]

        valid_mask_b1hw, src_depth_sampled_b1hw = self.get_valid_mask(
            cur_depth_b1hw,
            src_depth_b1hw,
            cur_invK_b44,
            src_K_b44,
            cur_world_T_cam_b44,
            src_cam_T_world_b44,
        )

        pred_cam_points_b4N = self.backproject(depth_pred_b1hw, cur_invK_b44)
        pred_world_points_b4N = cur_world_T_cam_b44 @ pred_cam_points_b4N

        src_cam_points_b3N = self.project(pred_world_points_b4N, src_K_b44, src_cam_T_world_b44)

        pred_cam_points_b3hw = src_cam_points_b3N.view(-1, 3, depth_height, depth_width)
        pred_src_depths_b1hw = pred_cam_points_b3hw[:, 2:]

        depth_diff_b1hw = torch.abs(
            torch.log(src_depth_sampled_b1hw) - torch.log(pred_src_depths_b1hw)
        ).masked_select(valid_mask_b1hw)

        depth_loss = depth_diff_b1hw.nanmean()

        return depth_loss

    def forward(
        self,
        depth_pred_b1hw,
        cur_depth_b1hw,
        src_depth_bk1hw,
        cur_invK_b44,
        src_K_bk44,
        cur_world_T_cam_b44,
        src_cam_T_world_bk44,
    ):
        src_to_iterate = [
            torch.unbind(src_depth_bk1hw, dim=1),
            torch.unbind(src_K_bk44, dim=1),
            torch.unbind(src_cam_T_world_bk44, dim=1),
        ]

        num_src_frames = src_depth_bk1hw.shape[1]

        loss = 0
        for src_depth_b1hw, src_K_b44, src_cam_T_world_b44 in zip(*src_to_iterate):
            error = self.get_error_for_pair(
                depth_pred_b1hw,
                cur_depth_b1hw,
                src_depth_b1hw,
                cur_invK_b44,
                src_K_b44,
                cur_world_T_cam_b44,
                src_cam_T_world_b44,
            )
            loss += error

        return loss / num_src_frames


class TemporalGradientMatchingLoss(nn.Module):
    """Temporal gradient matching loss of Video Depth Anything.

    DriveMVS addition. For frame pairs that are 1, 2, 4, ... steps apart the
    change of the prediction over time is compared with the change of the
    ground truth at the same pixel, |(p_t+s - p_t) - (g_t+s - g_t)|. Pixels
    whose ground truth changes a lot (occlusion boundaries, moving objects)
    are left out and the largest residuals are trimmed.

    The frames must be pixel aligned, i.e. share crop and flip.
    """

    def __init__(self, trim=0.2, num_scales=4, scale_decay=0.5, change_threshold=0.05):
        """
        Args:
            trim: fraction of the largest residuals that is ignored.
            num_scales: number of temporal strides (1, 2, 4, ...).
            scale_decay: weight multiplier per temporal stride.
            change_threshold: pixels whose ground truth changes by more than
                this fraction of the frame's value range are ignored.
        """
        super().__init__()
        self.trim = trim
        self.num_scales = num_scales
        self.scale_decay = scale_decay
        self.change_threshold = change_threshold

    def forward(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            prediction, target: (B, T, H, W).
            mask: (B, T, H, W) bool, True where target is valid.
        """
        num_frames = prediction.shape[1]
        zero = prediction.sum() * 0.0
        if num_frames < 2:
            return zero

        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        inf = torch.full_like(target, float("inf"))
        target_min = torch.where(mask, target, inf).amin(dim=(-2, -1), keepdim=True)
        target_max = torch.where(mask, target, -inf).amax(dim=(-2, -1), keepdim=True)
        threshold = (target_max - target_min).clamp_min(0) * self.change_threshold
        threshold = torch.nan_to_num(threshold, nan=0.0, posinf=0.0, neginf=0.0)

        total, count = zero, 0
        for scale in range(self.num_scales):
            stride = 2**scale
            if stride >= num_frames:
                break
            pred_change = prediction[:, stride:] - prediction[:, :-stride]
            target_change = target[:, stride:] - target[:, :-stride]
            valid = (
                mask[:, stride:]
                & mask[:, :-stride]
                & (target_change.abs() < threshold[:, :-stride])
            )
            if not valid.any():
                continue
            residual = (pred_change - target_change).abs()[valid]
            keep = max(1, int(residual.numel() * (1.0 - self.trim)))
            residual = torch.topk(residual, keep, largest=False).values
            total = total + residual.mean() * self.scale_decay**scale
            count += 1

        return total / count if count > 0 else zero
