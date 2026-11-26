import inspect
import logging
import os
import random

import numpy as np
import PIL.Image as pil
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from mvsanywhere.utils.generic_utils import (
    imagenet_normalize,
    read_image_file,
    readlines,
)
from mvsanywhere.utils.geometry_utils import pose_distance, rotz
from mvsanywhere.utils.prompt_utils import (
    sample_velodyne_sensor,
    simulate_scanline_prompt,
    simulate_velodyne_prompt,
)

logger = logging.getLogger(__name__)


class ClipConsistentRandomResizedCrop(transforms.RandomResizedCrop):
    """RandomResizedCrop whose parameters can be frozen for a whole clip.

    Dataset classes draw their crop with
    ``self.random_resize_crop.get_params(...)``. For video clips every
    reference frame has to be cropped identically, otherwise neighbouring
    frames are not pixel aligned and temporal supervision is meaningless.
    While sharing is enabled the first set of parameters drawn for an image
    size is returned for every later request with the same size.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shared_params = {}
        self._sharing = False

    def begin_clip(self):
        """Forget the crop of the previous clip."""
        self._shared_params = {}

    def share(self, enabled: bool):
        """Turn sharing on (reference frames) or off (independent crops)."""
        self._sharing = enabled

    def get_params(self, img, scale, ratio):  # pylint: disable=arguments-differ
        if not self._sharing:
            return transforms.RandomResizedCrop.get_params(img, scale, ratio)
        key = tuple(img.shape[-2:])
        if key not in self._shared_params:
            self._shared_params[key] = transforms.RandomResizedCrop.get_params(img, scale, ratio)
        return self._shared_params[key]


class GenericMVSDataset(Dataset):
    """
    Generic MVS dataset class. This class can be used as a base
    for different multi-view datasets.

    It houses the main __getitem__ function that will assemble a tuple of imgaes
    and their data.

    Tuples are read from a tuple file defined as
        tuple_info_file_location/{split}{mv_tuple_file_suffix}

    Each line in the tuple file should contain a scene id and frame ids for each
    frame in the tuple:

        scan_id frame_id_0 frame_id_1 ... frame_id_N-1

    where frame_id_0 is the reference image.

    DriveMVS additionally reads *clip* lines, which hold several tuples of the
    same scan that are consecutive in time, separated by ';':

        scan_id;ref_0 src src ...;ref_1 src src ...;...;ref_T-1 src src ...

    A plain tuple line is treated as a clip of length one, so __getitem__
    always returns tensors with a leading time dimension: (T, ...) for the
    reference frames and (T, K, ...) for their source frames.

    These will be loaded and stored in self.frame_tuples.

    If no tuple file suffix is provided, the dataset will only allow basic frame
    data loading from the split.

    Datasets that use this base class as a parent should modify base file load
    functions that do not have an implementation below.

    """

    # Where sparse depth prompts come from:
    #   "none"      - the dataset provides no prompts (all-zero prompts are returned),
    #   "simulated" - LiDAR is simulated from the dense ground-truth depth,
    #   "lidar"     - a real LiDAR projection is read by load_lidar_prompt().
    PROMPT_SOURCE = "none"
    # number of scan lines of the deterministic pattern used outside of training
    # when prompts are simulated.
    PROMPT_EVAL_NUM_LINES = 16

    def __init__(
        self,
        dataset_path,
        split,
        mv_tuple_file_suffix,
        include_full_res_depth=False,
        limit_to_scan_id=None,
        num_images_in_tuple=None,
        tuple_info_file_location=None,
        image_height=480,
        image_width=640,
        high_res_image_width=None,
        high_res_image_height=None,
        image_depth_ratio=2,
        shuffle_tuple=False,
        include_full_depth_K=False,
        include_high_res_color=False,
        pass_frame_id=False,
        skip_frames=None,
        skip_to_frame=None,
        verbose_init=True,
        image_resampling_mode=pil.BILINEAR,
        disable_flip=False,
        rotate_images=False,
        matching_scale=0.25,
        prediction_scale=0.5,
        prediction_num_scales=5,
    ):
        """
        Args:
            dataset_path: base path to the dataaset directory.
            split: the dataset split.
            mv_tuple_file_suffix: a suffix for the tuple file's name. The
                tuple filename searched for wil be
                {split}{mv_tuple_file_suffix}.
            tuple_info_file_location: location to search for a tuple file, if
                None provided, will search in the dataset directory under
                'tuples'.
            limit_to_scan_id: limit loaded tuples to one scan's frames.
            num_images_in_tuple: optional integer to limit tuples to this number
                of images.
            image_height, image_width: size images should be loaded at/resized
                to.
            include_high_res_color: should the dataset pass back higher
                resolution images.
            high_res_image_height, high_res_image_width: resolution images
                should be resized if we're passing back higher resolution
                images.
            image_depth_ratio: returned gt depth maps "depth_b1hw" will be of
                size (image_height, image_width)/image_depth_ratio.
            include_full_res_depth: if true will return depth maps from the
                dataset at the highest resolution available.
            shuffle_tuple: by default source images will be ordered according to
                overall pose distance to the reference image. When this flag is
                true, source images will be shuffled. Only used for ablation.
            pass_frame_id: if we should return the frame_id as part of the item
                dict
            skip_frames: if not none, will stride the tuple list by this value.
                Useful for only fusing every 'skip_frames' frame when fusing
                depth.
            verbose_init: if True will let the init print details on the
                initialization.
            native_depth_width, native_depth_height: for some datasets, it's
                useful to know what the native depth resolution is in advance.
            image_resampling_mode: resampling method for resizing images.

        """
        super(GenericMVSDataset).__init__()

        self.split = split
        scan_folder = self.get_sub_folder_dir(split)

        self.dataset_path = dataset_path
        self.scenes_path = os.path.join(dataset_path, scan_folder)

        self.mv_tuple_file_suffix = mv_tuple_file_suffix
        self.num_images_in_tuple = num_images_in_tuple
        self.shuffle_tuple = shuffle_tuple

        # default to where the dataset is to look for a tuple file
        if tuple_info_file_location is None:
            tuple_info_file_location = os.path.join(dataset_path, "tuples")

        if mv_tuple_file_suffix is not None:
            # tuple info should be available
            tuple_information_filepath = os.path.join(
                os.environ["PWD"], tuple_info_file_location, f"{split}{mv_tuple_file_suffix}"
            )

            # check if this file exists
            assert os.path.exists(tuple_information_filepath), (
                f"Tuple file {tuple_information_filepath} "
                "doesn't exist! Pass none for mv_tuple_file_suffix if you don't"
                " actually need a tuple file, otherwise check your paths."
            )

            # read in those tuples
            self.frame_tuples = readlines(tuple_information_filepath)

            # optionally limit frames to just one scan.
            if limit_to_scan_id is not None:
                self.frame_tuples = [
                    frame_tuple
                    for frame_tuple in self.frame_tuples
                    if limit_to_scan_id == self.parse_tuple(frame_tuple)[0]
                ]

            if skip_to_frame is not None:
                if verbose_init:
                    print(f"".center(80, "#"))
                    print(f"".center(80, "#"))
                    print(f"".center(80, "#"))
                    print(f" Skipping to frame {skip_to_frame} ".center(80, "#"))
                    print(f"".center(80, "#"))
                    print(f"".center(80, "#"))
                    print(f"".center(80, "#"), "\n")
                self.frame_tuples = self.frame_tuples[skip_to_frame:]

            # optionally skip every frame with interval skip_frame
            if skip_frames is not None:
                if verbose_init:
                    print(f"".center(80, "#"))
                    print(f"".center(80, "#"))
                    print(f"".center(80, "#"))
                    print(f" Skipping every {skip_frames} ".center(80, "#"))
                    print(f"".center(80, "#"))
                    print(f"".center(80, "#"))
                    print(f"".center(80, "#"), "\n")
                self.frame_tuples = self.frame_tuples[::skip_frames]
        else:
            if verbose_init:
                print(f"".center(80, "#"))
                print(
                    f" tuple_information_filepath isn't provided."
                    "Only basic dataloader functions are available. ".center(80, "#")
                )
                print(f"".center(80, "#"), "\n")

        self.image_width = image_width
        self.image_height = image_height
        self.high_res_image_width = high_res_image_width
        self.high_res_image_height = high_res_image_height

        # Random resize crop. Reference frames of one clip share their crop.
        self.random_resize_crop = ClipConsistentRandomResizedCrop(
            (self.image_height, self.image_width), scale=(0.75, 1.0), ratio=(4 / 3, 4 / 3)
        )

        # prompt options, see configure_prompts().
        self.prompt_source = self.PROMPT_SOURCE
        self.prompt_eval_num_lines = self.PROMPT_EVAL_NUM_LINES
        self.prompt_simulate_in_eval = False
        self.clip_consistent_prompt_sensor = True
        self._clip_prompt_sensor = None

        # size up depth using ratio of RGB to depth
        self.depth_height = int(self.image_height * prediction_scale)
        self.depth_width = int(self.image_width * prediction_scale)

        self.matching_height = int(self.image_height * matching_scale)
        self.matching_width = int(self.image_width * matching_scale)

        self.include_full_depth_K = include_full_depth_K
        self.include_high_res_color = include_high_res_color
        self.include_full_res_depth = include_full_res_depth

        self.pass_frame_id = pass_frame_id

        self.disable_resize_warning = False
        self.image_resampling_mode = image_resampling_mode

        self.disable_flip = disable_flip

        self.rotate_images = rotate_images

        self.matching_scale = matching_scale
        self.prediction_scale = prediction_scale
        self.prediction_num_scales = prediction_num_scales

        # If high resolution image size is not provided, 
        # we use the one from the first frame
        if self.include_high_res_color and (
            self.high_res_image_height is None or self.high_res_image_width is None
        ):
            self.high_res_image_height = None
            self.high_res_image_width = None
            first_scan, first_groups = self.parse_tuple(self.frame_tuples[0])
            first_image = self.load_high_res_color(first_scan, first_groups[0][0])
            self.high_res_image_height = first_image.shape[1]
            self.high_res_image_width = first_image.shape[2]

    def __len__(self):
        return len(self.frame_tuples)

    @staticmethod
    def get_sub_folder_dir(split):
        """Where scans are for each split."""
        return ""

    def get_valid_frame_path(self, split, scan):
        """returns the filepath of a file that contains valid frame ids for a
        scan."""

        raise NotImplementedError()

    def get_valid_frame_ids(self, split, scan, store_computed=True):
        """Either loads or computes the ids of valid frames in the dataset for
        a scan.

        A valid frame is one that has an existing RGB frame, an existing
        depth file, and existing pose file where the pose isn't inf, -inf,
        or nan.

        Args:
            split: the data split (train/val/test)
            scan: the name of the scan
            store_computed: store the valid_frame file where we'd expect to
            see the file in the scan folder. get_valid_frame_path defines
            where this file is expected to be. If the file can't be saved,
            a warning will be printed and the exception reason printed.

        Returns:
            valid_frames: a list of strings with info on valid frames.
            Each string is a concat of the scan_id and the frame_id.
        """
        raise NotImplementedError()

    def get_color_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's color file at the dataset's
        configured RGB resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Either the filepath for a precached RGB file at the size
            required, or if that doesn't exist, the full size RGB frame
            from the dataset.

        """
        raise NotImplementedError()

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's higher res color file at the
        dataset's configured high RGB resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Either the filepath for a precached RGB file at the high res
            size required, or if that doesn't exist, the full size RGB frame
            from the dataset.

        """

        raise NotImplementedError()

    def get_cached_depth_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's depth file at the dataset's
        configured depth resolution.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Filepath for a precached depth file at the size
            required.

        """
        raise NotImplementedError()

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's depth file at the native
        resolution in the dataset.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Either the filepath for a precached depth file at the size
            required, or if that doesn't exist, the full size depth frame
            from the dataset.

        """
        raise NotImplementedError()

    def get_pose_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's pose file.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            Filepath for pose information.

        """
        raise NotImplementedError()

    def get_frame_id_string(self, frame_id):
        """Returns an id string for this frame_id that's unique to this frame
        within the scan.

        This string is what this dataset uses as a reference to store files
        on disk.
        """
        raise NotImplementedError()

    def get_gt_mesh_path(dataset_path, split, scan_id):
        """
        Returns a path to a gt mesh reconstruction file.
        """
        raise NotImplementedError()

    def load_intrinsics(self, scan_id, frame_id=None, flip=None):
        """Loads intrinsics, computes scaled intrinsics, and returns a dict
        with intrinsics matrices for a frame at multiple scales.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame. Not needed for ScanNet as images
            share intrinsics across a scene.
            flip: flips intrinsics along x for flipped images.

        Returns:
            output_dict: A dict with
                - K_s{i}_b44 (intrinsics) and invK_s{i}_b44
                (backprojection) where i in [0,1,2,3,4]. i=0 provides
                intrinsics at the scale for depth_b1hw.
                - K_full_depth_b44 and invK_full_depth_b44 provides
                intrinsics for the maximum available depth resolution.
                Only provided when include_full_res_depth is true.

        """
        raise NotImplementedError()

    def load_target_size_depth_and_mask(self, scan_id, frame_id):
        """Loads a depth map at the resolution the dataset is configured for.

        Internally, if the loaded depth map isn't at the target resolution,
        the depth map will be resized on-the-fly to meet that resolution.

        NOTE: This function will place NaNs where depth maps are invalid.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            depth: depth map at the right resolution. Will contain NaNs
                where depth values are invalid.
            mask: a float validity mask for the depth maps. (1.0 where depth
            is valid).
            mask_b: like mask but boolean.
        """
        raise NotImplementedError()

    def load_full_res_depth_and_mask(self, scan_id, frame_id):
        """Loads a depth map at the native resolution the dataset provides.

        NOTE: This function will place NaNs where depth maps are invalid.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            full_res_depth: depth map at the right resolution. Will contain
                NaNs where depth values are invalid.
            full_res_mask: a float validity mask for the depth maps. (1.0
            where depth is valid).
            full_res_mask_b: like mask but boolean.
        """
        raise NotImplementedError()

    def load_pose(self, scan_id, frame_id):
        """Loads a frame's pose file.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            world_T_cam (numpy array): matrix for transforming from the
                camera to the world (pose).
            cam_T_world (numpy array): matrix for transforming from the
                world to the camera (extrinsics).

        """
        raise NotImplementedError()

    def load_color(self, scan_id, frame_id, crop=None):
        """Loads a frame's RGB file, resizes it to configured RGB size.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            iamge: tensor of the resized RGB image at self.image_height and
            self.image_width resolution.

        """
        color_filepath = self.get_color_filepath(scan_id, frame_id)
        try:
            image = read_image_file(
                color_filepath,
                height=self.image_height,
                width=self.image_width,
                resampling_mode=self.image_resampling_mode,
                disable_warning=True,
                crop=crop,
            )
        except:
            print("Failed to load: ", scan_id, frame_id)
            image = torch.zeros((3, self.image_height, self.image_width)).float()

        # Remove alpha channel for PNGs
        image = image[:3]

        return image

    def load_high_res_color(self, scan_id, frame_id):
        """Loads a frame's RGB file at a high resolution as configured.

        Args:
            scan_id: the scan this file belongs to.
            frame_id: id for the frame.

        Returns:
            iamge: tensor of the resized RGB image at
            self.high_res_image_height and self.high_res_image_width
            resolution.

        """

        color_high_res_filepath = self.get_high_res_color_filepath(scan_id, frame_id)
        high_res_color = read_image_file(
            color_high_res_filepath,
            height=self.high_res_image_height,
            width=self.high_res_image_width,
            resampling_mode=self.image_resampling_mode,
            disable_warning=self.disable_resize_warning,
        )

        # Remove alpha channel for PNGs
        high_res_color = high_res_color[:3]

        return high_res_color

    # ------------------------------------------------------------------ #
    # sparse depth prompts (DriveMVS)
    # ------------------------------------------------------------------ #
    def configure_prompts(
        self,
        prompt_source=None,
        eval_num_lines=None,
        simulate_in_eval=None,
        clip_consistent_sensor=None,
        **dataset_specific,
    ):
        """Overrides how sparse depth prompts are produced.

        Arguments left as None keep the defaults of the dataset class.

        Args:
            prompt_source: "none", "simulated" or "lidar", see PROMPT_SOURCE.
            eval_num_lines: number of scan lines of the deterministic pattern
                used outside of training when prompts are simulated.
            simulate_in_eval: use the random training-time LiDAR simulator
                outside of training as well.
            clip_consistent_sensor: keep one simulated sensor for all frames
                of a clip instead of drawing a new one for every frame.
            dataset_specific: options understood by a dataset class, e.g. the
                number of LiDAR beams to keep. Unknown names raise an error.
        """
        if prompt_source is not None:
            assert prompt_source in ("none", "simulated", "lidar")
            self.prompt_source = prompt_source
        if eval_num_lines is not None:
            self.prompt_eval_num_lines = int(eval_num_lines)
        if simulate_in_eval is not None:
            self.prompt_simulate_in_eval = bool(simulate_in_eval)
        if clip_consistent_sensor is not None:
            self.clip_consistent_prompt_sensor = bool(clip_consistent_sensor)
        for name, value in dataset_specific.items():
            if value is None:
                continue
            if not hasattr(self, name):
                raise ValueError(f"{type(self).__name__} has no prompt option '{name}'")
            setattr(self, name, value)
        return self

    def load_lidar_prompt(self, scan_id, frame_id, intrinsics, crop, flip):
        """Loads a real LiDAR prompt for a frame.

        Only needed by datasets with PROMPT_SOURCE == "lidar".

        Returns:
            prompt: (1, depth_height, depth_width) sparse depth in metres, 0
                where there is no measurement, already cropped / flipped like
                the image.
        """
        raise NotImplementedError()

    def load_full_res_lidar_prompt(self, scan_id, frame_id, flip):
        """Optionally returns the LiDAR prompt at the native depth resolution,
        (1, H, W) with 0 where there is no measurement, or None."""
        return None

    def load_sparse_prompt(self, scan_id, frame_id, depth, intrinsics, crop, flip):
        """Returns the sparse depth prompt of a frame.

        Args:
            depth: ground-truth depth at the target resolution (1, h, w) with
                NaNs where invalid, or None if it was not loaded.

        Returns:
            prompt: (1, depth_height, depth_width) sparse depth in metres, 0
                where there is no measurement.
        """
        if self.prompt_source == "lidar":
            return self.load_lidar_prompt(scan_id, frame_id, intrinsics, crop, flip).float()

        if self.prompt_source == "simulated" and depth is not None:
            dense = torch.where(torch.isfinite(depth[0]), depth[0], torch.zeros_like(depth[0]))
            if self.split == "train" or self.prompt_simulate_in_eval:
                sensor = self._clip_prompt_sensor if self.clip_consistent_prompt_sensor else None
                prompt = simulate_velodyne_prompt(dense, sensor)
            else:
                prompt = simulate_scanline_prompt(dense, self.prompt_eval_num_lines)
            return prompt.unsqueeze(0).float()

        return torch.zeros((1, self.depth_height, self.depth_width), dtype=torch.float32)

    def get_frame(self, scan_id, frame_id, load_depth, flip=False, load_prompt=False):
        """Retrieves a single frame's worth of information.

        NOTE: Returned depth maps will use NaN for values where the depth
        map is invalid.

        Args:
            scan_id: a string defining the scan this frame belongs to.
            frame_id: an integer id for this frame.
            load_depth: a bool flag for loading depth maps and not dummy
                data
            flip: flips images, depth maps, and intriniscs along x.
        Returns:
            output_dict: a dictionary with this frame's information,
            including:
             - image_b3hw: an imagenet normalized RGB tensor of the image,
                resized to [self.image_height, self.image_width].
             - depth_b1hw: groundtruth depth map for this frame tensor,
                resized to [self.depth_height, self.depth_width.]
             - mask_b1hw: valid float mask where 1.0 indicates a valid depth
                value in depth_b1hw.
             - mask_b_b1hw: like mask_b1hw but binary.
             - world_T_cam_b44: transform for transforming points from
                camera to world coordinates. (pose)
             - cam_T_world_b44: transform for transforming points from world
                to camera coordinaetes. (extrinsics)
             - intrinsics: a dictionary with intrinsics at various
                resolutions and their inverses. Includes:
                    - K_s{i}_b44 (intrinsics) and invK_s{i}_b44
                    (backprojection) where i in [0,1,2,3,4]. i=0 provides
                    intrinsics at the scale for depth_b1hw.
                    - K_full_depth_b44 and invK_full_depth_b44 provides
                    intrinsics for the maximum available depth resolution.
                    Only provided when include_full_res_depth is true.
             - frame_id_string: a string that uniquly identifies the frame
                as it is on disk in its filename. Provided when
                pass_frame_id is true.
             - high_res_color_b3hw: an imagenet normalized RGB tensor of the
                image, at 640 (w) by 480 (h) resolution.
                Provided when include_high_res_color is true.
             - full_res_depth_b1hw: highest resolution depth map available.
                Will only be available if include_full_res_depth is true.
                Provided when include_full_res_depth is true.
             - full_res_mask_b1hw: valid float mask where 1.0 indicates a
                valid depth value in full_res_depth_b1hw.
                Provided when include_full_res_depth is true.
             - full_res_mask_b_b1hw: like full_res_mask_b1hw but binary.
             - min_depth: minimum depth in the gt
             - max_depth: maximum depth value in the gt
             - sparse_prompt_b1hw: sparse metric depth prompt at the depth
                resolution, 0 where there is no measurement. Provided when
                load_prompt is true.
             - sparse_prompt_mask_b1hw: float validity mask of the prompt.

        """
        # stores output
        output_dict = {}
        depth = None

        # load pose
        world_T_cam, cam_T_world = self.load_pose(scan_id, frame_id)

        # load intrinsics
        intrinsics, crop = self.load_intrinsics(scan_id, frame_id, flip=flip)

        if self.rotate_images:
            T = np.eye(4)
            T[:3, :3] = rotz(-np.pi / 2)
            world_T_cam = world_T_cam @ T
            cam_T_world = np.linalg.inv(world_T_cam)

        if flip:
            T = np.eye(4).astype(world_T_cam.dtype)
            T[0, 0] = -1.0
            world_T_cam = world_T_cam @ T
            cam_T_world = np.linalg.inv(world_T_cam)

        # Load image
        image = self.load_color(scan_id, frame_id, crop)

        if self.rotate_images:
            image = torch.rot90(image, 3, [1, 2])

        if flip:
            image = torch.flip(image, (-1,))

        # Do imagenet normalization
        image = imagenet_normalize(image)

        output_dict.update(
            {
                "image_b3hw": image,
                "world_T_cam_b44": world_T_cam,
                "cam_T_world_b44": cam_T_world,
            }
        )

        output_dict.update(intrinsics)

        if load_depth:
            # get depth
            depth_outputs = self.load_target_size_depth_and_mask(scan_id, frame_id, crop)
            if len(depth_outputs) == 3:
                depth, mask, mask_b = depth_outputs
                skymask = torch.full_like(mask, torch.nan)
            else:
                depth, mask, mask_b, skymask = depth_outputs
                skymask = skymask.float()

            if self.rotate_images:
                depth = torch.rot90(depth, 3, [1, 2])
                mask = torch.rot90(mask, 3, [1, 2])
                mask_b = torch.rot90(mask_b, 3, [1, 2])
                skymask = torch.rot90(skymask, 3, [1, 2])

            if flip:
                depth = torch.flip(depth, (-1,))
                mask = torch.flip(mask, (-1,))
                mask_b = torch.flip(mask_b, (-1,))
                skymask = torch.flip(skymask, (-1,))

            max_depth = depth[torch.isfinite(depth)].max() if torch.isfinite(depth).any().item() else torch.tensor(10.0)
            min_depth = depth[torch.isfinite(depth)].min() if torch.isfinite(depth).any().item() else torch.tensor(10.0)

            max_depth = max_depth * (torch.rand(1)[0] + 1.0)
            min_depth = min_depth * (torch.rand(1)[0] * 0.5 + 0.5)

            output_dict.update(
                {
                    "depth_b1hw": depth,
                    "mask_b1hw": mask,
                    "mask_b_b1hw": mask_b,
                    "max_depth": max_depth, #depth[torch.isfinite(depth)].max() if torch.isfinite(depth).any().item() else torch.tensor(10.0),
                    "min_depth": min_depth, #depth[torch.isfinite(depth)].min() if torch.isfinite(depth).any().item() else torch.tensor(1.0),
                    "skymask_b1hw": skymask,
                }
            )

        # Load high res image
        if self.include_high_res_color:
            high_res_color = self.load_high_res_color(scan_id, frame_id)
            high_res_color = imagenet_normalize(high_res_color)

            if self.rotate_images:
                high_res_color = torch.rot90(high_res_color, 3, [1, 2])

            if flip:
                high_res_color = torch.flip(high_res_color, (-1,))

            output_dict.update(
                {
                    "high_res_color_b3hw": high_res_color,
                }
            )

        if load_prompt:
            if self.rotate_images and self.prompt_source != "none":
                raise NotImplementedError("Prompts are not supported together with rotate_images.")
            prompt = self.load_sparse_prompt(scan_id, frame_id, depth, intrinsics, crop, flip)
            output_dict.update(
                {
                    "sparse_prompt_b1hw": prompt,
                    "sparse_prompt_mask_b1hw": (prompt > 0).float(),
                }
            )

        if self.include_full_res_depth and load_depth:
            # get high res depth. Not every dataset class takes the crop.
            if "crop" in inspect.signature(self.load_full_res_depth_and_mask).parameters:
                full_res_outputs = self.load_full_res_depth_and_mask(scan_id, frame_id, crop)
            else:
                full_res_outputs = self.load_full_res_depth_and_mask(scan_id, frame_id)
            full_res_depth, full_res_mask, full_res_mask_b = full_res_outputs

            if self.rotate_images:
                full_res_depth = torch.rot90(full_res_depth, 3, [1, 2])
                full_res_mask = torch.rot90(full_res_mask, 3, [1, 2])
                full_res_mask_b = torch.rot90(full_res_mask_b, 3, [1, 2])

            if flip:
                full_res_depth = torch.flip(full_res_depth, (-1,))
                full_res_mask = torch.flip(full_res_mask, (-1,))
                full_res_mask_b = torch.flip(full_res_mask_b, (-1,))

            output_dict.update(
                {
                    "full_res_depth_b1hw": full_res_depth,
                    "full_res_mask_b1hw": full_res_mask,
                    "full_res_mask_b_b1hw": full_res_mask_b,
                }
            )

            # Real LiDAR prompts at the native resolution, so that evaluation
            # can separate ground-truth pixels that carry a prompt sample
            # from those that do not.
            if load_prompt and self.prompt_source == "lidar":
                full_res_prompt = self.load_full_res_lidar_prompt(scan_id, frame_id, flip)
                if full_res_prompt is not None:
                    output_dict["full_res_prompt_b1hw"] = full_res_prompt.float()

        if self.pass_frame_id:
            output_dict["frame_id_string"] = self.get_frame_id_string(frame_id)

        return output_dict

    def stack_src_data(self, src_data):
        """Stacks source image data into tensors."""

        tensor_names = src_data[0].keys()
        stacked_src_data = {}
        for tensor_name in tensor_names:
            if "frame_id_string" in tensor_name:
                stacked_src_data[tensor_name] = [t[tensor_name] for t in src_data]
            else:
                stacked_src_data[tensor_name] = np.stack([t[tensor_name] for t in src_data], axis=0)

        return stacked_src_data

    @staticmethod
    def stack_clip_data(frame_data_list):
        """Stacks the per-time-step dictionaries of a clip along a new first
        (time) dimension."""
        stacked = {}
        for key in frame_data_list[0].keys():
            if "frame_id_string" in key:
                stacked[key] = [d[key] for d in frame_data_list]
            else:
                stacked[key] = np.stack([np.asarray(d[key]) for d in frame_data_list], axis=0)
        return stacked

    @staticmethod
    def parse_tuple(line):
        """Parses one line of a tuple file.

        Returns:
            scan_id: the scan the line belongs to.
            frame_groups: one list of frame ids (reference first) per time
                step. Plain tuple lines yield a single group.
        """
        line = line.strip()
        if ";" in line:
            scan_id, *groups = line.split(";")
            frame_groups = [group.split() for group in groups if group.strip()]
        else:
            scan_id, *frame_ids = line.split()
            frame_groups = [frame_ids]
        return scan_id.strip(), frame_groups

    def source_frames_need_depth(self):
        """Ground-truth depth of source frames is only read when their
        prompts are simulated from it."""
        return self.prompt_source == "simulated"

    def __getitem__(self, idx):
        """Loads data for all frames of the clip (or single MVS tuple) at
        index idx.

        Args:
            idx: the index for the elmeent in the dataset.

        Returns:
            cur_data: frame data for the T reference frames, every tensor has
                a leading time dimension (T, ...).
            src_data: stacked frame data for the K source frames of every
                reference frame, (T, K, ...).
        """

        flip_threshold = 0.5 if self.split == "train" and not self.disable_flip else 0.0
        flip = torch.rand(1).item() < flip_threshold

        scan_id, frame_groups = self.parse_tuple(self.frame_tuples[idx])

        # augmentations that have to agree across the clip
        self.random_resize_crop.begin_clip()
        self._clip_prompt_sensor = (
            sample_velodyne_sensor(self.depth_height, self.depth_width)
            if self.prompt_source == "simulated"
            else None
        )

        all_cur_data, all_src_data = [], []
        for frame_ids in frame_groups:
            # shuffle tuple order, by default false
            if self.shuffle_tuple:
                first_frame_id = frame_ids[0]
                shuffled_list = frame_ids[1:]
                random.shuffle(shuffled_list)
                frame_ids = [first_frame_id] + shuffled_list

            # the tuple file may have more images in the tuple than what might
            # be requested, so limit the tuple length to num_images_in_tuple
            if self.num_images_in_tuple is not None:
                frame_ids = frame_ids[: self.num_images_in_tuple]

            # reference frames of a clip share one crop so that they stay
            # pixel aligned in time; source frames keep independent crops.
            self.random_resize_crop.share(True)
            cur_data = self.get_frame(
                scan_id, frame_ids[0], load_depth=True, flip=flip, load_prompt=True
            )
            self.random_resize_crop.share(False)
            src_data_list = [
                self.get_frame(
                    scan_id,
                    frame_id,
                    load_depth=self.source_frames_need_depth(),
                    flip=flip,
                    load_prompt=True,
                )
                for frame_id in frame_ids[1:]
            ]

            # now sort all source frames according to pose penalty w.r.t to
            # the refernce frame (cur_data)
            if not self.shuffle_tuple:
                src_world_T_cam = torch.tensor(
                    np.stack([src["world_T_cam_b44"] for src in src_data_list], axis=0)
                )
                cur_cam_T_world = torch.tensor(cur_data["cam_T_world_b44"])

                # Compute cur_cam_T_src_cam
                cur_cam_T_src_cam = cur_cam_T_world.unsqueeze(0) @ src_world_T_cam

                # get penalties.
                frame_penalty_k, _, _ = pose_distance(cur_cam_T_src_cam)

                # order based on indices
                indices = torch.argsort(frame_penalty_k).tolist()
                src_data_list = [src_data_list[index] for index in indices]

            all_cur_data.append(cur_data)
            all_src_data.append(self.stack_src_data(src_data_list))

        return self.stack_clip_data(all_cur_data), self.stack_clip_data(all_src_data)
