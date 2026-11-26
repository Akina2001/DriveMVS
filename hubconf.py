dependencies = ['torch']

from pathlib import Path
from typing import Optional, Union

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms as T

import mvsanywhere.options as options
from mvsanywhere.utils.model_utils import get_model_class, load_model_inference


class MVSAnywhereInference(nn.Module):

    def __init__(
        self,
        pretrained: bool = True,
        dot_model: bool = False,
    ):
        """
        Initialize the MVSAnywhereInference model.
        Args:
            pretrained (bool): If True, loads the pre-trained weights.
            dot_model (bool): If True, uses dot product cost volume instead of metadata-based one.
        """
        super().__init__()

        # Create and load options
        option_handler = options.OptionsHandler()

        config_filepath = (
            Path(__file__).parent / 
            'configs' /
            'models' /
            ('mvsanywhere_model.yaml' if not dot_model else 'mvsanywhere_dot_model.yaml')
        )
        option_handler.parse_and_merge_options(config_filepaths=str(config_filepath))
        opts = option_handler.options

        # Download checkpoint if not exists
        if pretrained:
            ckpt_name = 'mvsanywhere_hero.ckpt' if not dot_model else 'mvsanywhere_dot.ckpt'
            torch.hub.download_url_to_file(
                f'https://storage.googleapis.com/niantic-lon-static/research/mvsanywhere/{ckpt_name}', 
                ckpt_name,
            )
            opts.load_weights_from_checkpoint = ckpt_name 

        model_class_to_use = get_model_class(opts)
        model = load_model_inference(opts, model_class_to_use)
        self.model = model.eval()

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.input_transform = T.Compose([
            T.ToTensor(), self.normalize
        ])

    def forward(
        self,
        cur_image: torch.Tensor,
        src_image: torch.Tensor,
        cur_pose: torch.Tensor,
        src_pose: torch.Tensor,
        cur_intrinsics: torch.Tensor,
        src_intrinsics: torch.Tensor,
        depth_range: Optional[torch.Tensor] = None,
        num_refinement_steps: int = 1,
    ):
        """
        Forward pass for the MVSAnywhere model with preprocessed tensor inputs.

        This function expects all inputs to be PyTorch tensors, properly batched and normalized.
        It performs the core depth estimation and refinement steps.

        Args:
            cur_image (torch.Tensor):
                Current/reference images tensor of shape [B, 3, H, W].
                Images must be normalized (e.g., mean/std normalized).

            src_image (torch.Tensor):
                Source images tensor of shape [B, N, 3, H, W].
                Must be normalized consistently with `cur_image`.

            cur_pose (torch.Tensor):
                Current camera poses tensor of shape [B, 4, 4].
                Each pose is a world-from-camera transformation matrix.

            src_pose (torch.Tensor):
                Source camera poses tensor of shape [B, N, 4, 4].
                Each pose is a world-from-camera transformation matrix.

            cur_intrinsics (torch.Tensor):
                Intrinsic matrices for the current images of shape [B, 4, 4].
                K matrix at the image resolution of cur_image.

            src_intrinsics (torch.Tensor):
                Intrinsic matrices for the source images of shape [B, N, 4, 4].
                K matrices at the image resolution of src_image.

            depth_range (torch.Tensor, optional):
                Optional depth range (min, max) per batch element of shape [B, 2].
                If not provided, depth range is inferred heuristically.

            num_refinement_steps (int, optional):
                Number of iterative refinement steps. Defaults to 1.

        Returns:
            dict: Dictionary containing depth predictions and confidence/masks:
                - 'depth' (torch.Tensor): Predicted depth maps of shape [B, 1, H, W].
                - 'confidence' (torch.Tensor, optional): Confidence or mask map aligned with depth.
        """

        cur_K_matching = cur_intrinsics.clone()
        cur_K_matching[:, :2] /= 4

        src_K_matching = src_intrinsics.clone()
        src_K_matching[:, :, :2] /= 4
 

        # Reference image data
        cur_data = {
            "image_b3hw": cur_image,
            "cam_T_world_b44": torch.linalg.inv(cur_pose),
            "world_T_cam_b44": cur_pose,
            "K_matching_b44": cur_K_matching,
            "invK_matching_b44": torch.linalg.inv(cur_K_matching),
        }

        # Multi-view source images
        src_data = {
            "image_b3hw": src_image,
            "cam_T_world_b44": torch.linalg.inv(src_pose),
            "world_T_cam_b44": src_pose,
            "K_matching_b44": src_K_matching,
            "invK_matching_b44": torch.linalg.inv(src_K_matching),
        }

        # If depth range is provided, add it to the current data
        if depth_range is not None:
            cur_data["min_depth"] = depth_range[:, 0]
            cur_data["max_depth"] = depth_range[:, 1]

        # Run the model!
        outputs = self.model(
            phase="test",
            cur_data=cur_data,
            src_data=src_data,
            return_mask=True,
            num_refinement_steps=num_refinement_steps,
            estimate_depth_range=depth_range is None,
        )

        return outputs
    

    def preprocess_and_run(
        self,
        cur_image: Union[Image.Image, np.ndarray],
        src_image: Union[list[Image.Image], list[np.ndarray]],
        cur_pose: np.ndarray,
        src_pose: np.ndarray,
        cur_intrinsics: np.ndarray,
        src_intrinsics: np.ndarray,
        depth_range: Optional[np.ndarray] = None,
        num_refinement_steps: int = 1,
        target_image_size: Optional[tuple[int, int]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Preprocess input images, poses, and intrinsics from PIL images or numpy arrays,
        then run the MVSAnywhere model forward pass.

        This function handles:
            - Conversion of PIL Images or numpy arrays to PyTorch tensors.
            - Normalization of image pixel values (e.g., scaling to [0,1] and mean/std normalization).
            - Adding batch dimensions if needed.
            - Moving all tensors to the model's device.
            - Passing the processed tensors to `forward()`.

        Args:
            cur_image (PIL.Image.Image | np.ndarray):
                Current/reference image.
                PIL images or numpy arrays should be in shape [H, W, 3] with pixel values [0, 255].

            src_image (list[PIL.Image.Image | np.ndarray] | np.ndarray):
                Source images corresponding to each current image.
                List with each item being a PIL Image or numpy array of shape [H, W, 3].

            cur_pose (np.ndarray):
                Current camera poses of shape [4, 4].
                Each pose is a world-from-camera transformation matrix.

            src_pose (np.ndarray):
                Source camera poses of shape [N, 4, 4].
                Each pose is a world-from-camera transformation matrix.

            cur_intrinsics (np.ndarray):
                Intrinsic matrices for current images of shape [4, 4].
                K matrix at the image resolution of cur_image.

            src_intrinsics (np.ndarray):
                Intrinsic matrices for source images of shape [N, 4, 4].
                K matrices at the image resolution of cur_image.

            depth_range (np.ndarray | None, optional):
                Optional depth range per batch element of shape [2].

            num_refinement_steps (int, optional):
                Number of iterative refinement steps. Defaults to 1.

            target_image_size (tuple[int, int], optional):
                If provided, resizes the input images to this size [H, W] before processing.
                Intrinsics will be adjusted accordingly.

            device (torch.device | None, optional):
                Device to run the model on.

        Returns:
            dict: Dictionary with the same keys and values as returned by `forward()`.
        """
        if target_image_size is not None:
            resize_transform = T.Resize(target_image_size)

            # Adjust intrinsics
            if isinstance(cur_image, np.ndarray):
                H, W = cur_image.shape[:2]
            else:
                H, W = cur_image.size[::-1]
            scale_x = target_image_size[1] / W
            scale_y = target_image_size[0] / H
            cur_intrinsics[0] *= scale_x
            cur_intrinsics[1] *= scale_y
            src_intrinsics[:, 0] *= scale_x
            src_intrinsics[:, 1] *= scale_y
        else:
            resize_transform = T.Lambda(lambda x: x)

        cur_image = resize_transform(self.input_transform(cur_image))
        cur_image = cur_image.unsqueeze(0)

        src_image = [resize_transform(self.input_transform(img)) for img in src_image]
        src_image = torch.stack(src_image)
        src_image = src_image.unsqueeze(0)

        cur_pose = torch.tensor(cur_pose, dtype=torch.float32).unsqueeze(0)
        src_pose = torch.tensor(src_pose, dtype=torch.float32).unsqueeze(0)

        cur_intrinsics = torch.tensor(cur_intrinsics, dtype=torch.float32).unsqueeze(0)
        src_intrinsics = torch.tensor(src_intrinsics, dtype=torch.float32).unsqueeze(0)

        if depth_range is not None:
            depth_range = torch.tensor(depth_range, dtype=torch.float32).unsqueeze(0)

        outputs = self(
            cur_image=cur_image.to(device),
            src_image=src_image.to(device),
            cur_pose=cur_pose.to(device),
            src_pose=src_pose.to(device),
            cur_intrinsics=cur_intrinsics.to(device),
            src_intrinsics=src_intrinsics.to(device),
            depth_range=depth_range.to(device) if depth_range is not None else None,
            num_refinement_steps=num_refinement_steps,
        )

        return {
            'depth': outputs['depth_pred_s0_b1hw'].squeeze(),
            'sky_mask': outputs['sky_pred_s0_b1hw'].squeeze(),
            'depth_masked': outputs['depth_masked_pred_s0_b1hw'].squeeze(),
            'cost_volume': outputs['lowest_cost_bhw'].squeeze(),
        }



def mvsanywhere(
    pretrained: bool = True,
    dot_model: bool = False,
) -> torch.nn.Module:
    """Return a MVSAnywhere model.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on the complete MVSAnywhere training set.
        dot_model (bool): If True, uses dot product cost volume instead of metadata-based one
    Return:
        model (torch.nn.Module): the model.
    """

    model = MVSAnywhereInference(pretrained=pretrained, dot_model=dot_model)
    return model