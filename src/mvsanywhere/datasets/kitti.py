"""KITTI (raw + depth completion) as an MVS dataset with LiDAR prompts.

DriveMVS addition; not part of the upstream MVSAnywhere code base.

Expected directory layout (everything except ``drivemvs_meta`` is exactly what
the official KITTI archives extract to, see docs/data_preparation.md):

    <dataset_path>/
        raw_data/<date>/<date>_drive_<id>_sync/image_02/data/<frame>.png
        raw_data/<date>/<date>_drive_<id>_sync/oxts/data/<frame>.txt
        raw_data/<date>/calib_cam_to_cam.txt, calib_velo_to_cam.txt, ...
        <split>/<date>_drive_<id>_sync/proj_depth/groundtruth/image_02/<frame>.png
        <split>/<date>_drive_<id>_sync/proj_depth/velodyne_raw/image_02/<frame>.png
        drivemvs_meta/<date>_drive_<id>_sync/image_02/poses.txt
        drivemvs_meta/<date>_drive_<id>_sync/image_02/intrinsics.txt
        drivemvs_meta/<date>_drive_<id>_sync/image_02/sky_mask/<frame>.png   (optional)

``drivemvs_meta`` is written by ``tools/data/prepare_kitti.py``. A scan id is
``<split>/<date>_drive_<id>_sync/image_02``.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from mvsanywhere.datasets.generic_mvs_dataset import GenericMVSDataset
from mvsanywhere.utils.prompt_utils import resize_sparse_depth, sample_lidar_lines


class KITTIDataset(GenericMVSDataset):
    """
    MVS KITTI Dataset class.

    Inherits from GenericMVSDataset and implements missing methods. See
    GenericMVSDataset for how tuples work.

    Ground truth is the semi-dense accumulated LiDAR depth of the KITTI depth
    benchmark. Prompts are the raw single-sweep LiDAR projection, thinned to
    fewer scan lines.

    NOTE: This dataset will place NaNs where gt depth maps are invalid.

    """

    PROMPT_SOURCE = "lidar"

    def __init__(
        self,
        dataset_path,
        split,
        mv_tuple_file_suffix,
        include_full_res_depth=False,
        limit_to_scan_id=None,
        num_images_in_tuple=None,
        tuple_info_file_location=None,
        image_height=384,
        image_width=512,
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
        min_valid_depth=1e-2,
        max_valid_depth=1e2,
        disable_flip=False,
        rotate_images=False,
        matching_scale=0.25,
        prediction_scale=0.5,
        prediction_num_scales=5,
    ):
        # LiDAR prompt options, can be changed with configure_prompts().
        # The Velodyne HDL-64E has 64 beams; keeping every 2nd elevation bin
        # approximates a 32 beam sensor, every 4th a 16 beam sensor, ...
        self.lidar_num_bins = 64
        self.lidar_keep_every = 2
        # optional fixed (min, max) elevation in radians for the binning.
        self.lidar_pitch_range = None

        self._pose_cache = {}
        self._intrinsics_cache = {}

        super().__init__(
            dataset_path=dataset_path,
            split=split,
            mv_tuple_file_suffix=mv_tuple_file_suffix,
            include_full_res_depth=include_full_res_depth,
            limit_to_scan_id=limit_to_scan_id,
            num_images_in_tuple=num_images_in_tuple,
            tuple_info_file_location=tuple_info_file_location,
            image_height=image_height,
            image_width=image_width,
            high_res_image_width=high_res_image_width,
            high_res_image_height=high_res_image_height,
            image_depth_ratio=image_depth_ratio,
            shuffle_tuple=shuffle_tuple,
            include_full_depth_K=include_full_depth_K,
            include_high_res_color=include_high_res_color,
            pass_frame_id=pass_frame_id,
            skip_frames=skip_frames,
            skip_to_frame=skip_to_frame,
            verbose_init=verbose_init,
            disable_flip=disable_flip,
            rotate_images=rotate_images,
            matching_scale=matching_scale,
            prediction_scale=prediction_scale,
            prediction_num_scales=prediction_num_scales,
        )

        self.min_valid_depth = min_valid_depth
        self.max_valid_depth = max_valid_depth

    # ------------------------------------------------------------------ #
    # paths
    # ------------------------------------------------------------------ #
    @staticmethod
    def split_scan_id(scan_id):
        """'val/2011_09_26_drive_0002_sync/image_02' -> (split, drive, camera, date)."""
        split, drive, camera = scan_id.split("/")
        return split, drive, camera, drive.split("_drive")[0]

    def get_meta_dir(self, scan_id):
        _, drive, camera, _ = self.split_scan_id(scan_id)
        return Path(self.dataset_path) / "drivemvs_meta" / drive / camera

    def get_frame_id_string(self, frame_id):
        """Returns an id string for this frame_id that's unique to this frame
        within the scan."""
        return f"{int(frame_id):010d}"

    def get_valid_frame_path(self, split, scan):
        """returns the filepath of a file that contains valid frame ids for a
        scan."""
        return str(self.get_meta_dir(scan) / "valid_frames.txt")

    def get_valid_frame_ids(self, split, scan, store_computed=True):
        """Either loads or computes the ids of valid frames in the dataset for
        a scan. A valid frame has an RGB image, a ground-truth depth map and a
        finite pose.

        Returns:
            valid_frames: a list of strings "scan frame_id dist_to_last_valid".
        """
        valid_frame_path = self.get_valid_frame_path(split, scan)

        if os.path.exists(valid_frame_path):
            with open(valid_frame_path) as f:
                return f.readlines()

        print(f"computing valid frames for scene {scan}.")
        num_frames = self._load_poses(scan).shape[0]

        valid_frames = []
        dist_to_last_valid_frame = 0
        for frame_id in range(num_frames):
            pose_ok = np.isfinite(self._load_poses(scan)[frame_id]).all()
            if (
                os.path.exists(self.get_color_filepath(scan, frame_id))
                and os.path.exists(self.get_full_res_depth_filepath(scan, frame_id))
                and pose_ok
            ):
                valid_frames.append(f"{scan} {frame_id} {dist_to_last_valid_frame}")
                dist_to_last_valid_frame = 0
            else:
                dist_to_last_valid_frame += 1

        if store_computed:
            try:
                with open(valid_frame_path, "w") as f:
                    f.write("\n".join(valid_frames) + "\n")
            except Exception as e:
                print(f"Couldn't save valid_frames at {valid_frame_path}, cause:\n{e}")

        return valid_frames

    def get_color_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's color file."""
        _, drive, camera, date = self.split_scan_id(scan_id)
        return str(
            Path(self.dataset_path)
            / "raw_data"
            / date
            / drive
            / camera
            / "data"
            / f"{int(frame_id):010d}.png"
        )

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """returns the filepath for a frame's higher res color file."""
        return self.get_color_filepath(scan_id, frame_id)

    def get_cached_depth_filepath(self, scan_id, frame_id):
        """We do not use cached depths in this dataset."""
        return ""

    def _proj_depth_filepath(self, scan_id, frame_id, kind):
        split, drive, camera, _ = self.split_scan_id(scan_id)
        return str(
            Path(self.dataset_path)
            / split
            / drive
            / "proj_depth"
            / kind
            / camera
            / f"{int(frame_id):010d}.png"
        )

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """returns the filepath of the semi-dense ground-truth depth."""
        return self._proj_depth_filepath(scan_id, frame_id, "groundtruth")

    def get_full_res_prompt_filepath(self, scan_id, frame_id):
        """returns the filepath of the raw single-sweep LiDAR projection."""
        return self._proj_depth_filepath(scan_id, frame_id, "velodyne_raw")

    def get_sky_mask_filepath(self, scan_id, frame_id):
        """returns the filepath of an optional sky mask (white = sky)."""
        return str(self.get_meta_dir(scan_id) / "sky_mask" / f"{int(frame_id):010d}.png")

    # ------------------------------------------------------------------ #
    # calibration and poses
    # ------------------------------------------------------------------ #
    def _load_poses(self, scan_id):
        if scan_id not in self._pose_cache:
            poses = np.loadtxt(self.get_meta_dir(scan_id) / "poses.txt").reshape(-1, 3, 4)
            self._pose_cache[scan_id] = poses.astype(np.float64)
        return self._pose_cache[scan_id]

    def _load_native_intrinsics(self, scan_id):
        """Returns the 3x3 intrinsics and the native (height, width)."""
        if scan_id not in self._intrinsics_cache:
            K = np.loadtxt(self.get_meta_dir(scan_id) / "intrinsics.txt")[:3, :3]
            num_frames = self._load_poses(scan_id).shape[0]
            # image size differs between recording days, read it from an image.
            for frame_id in range(num_frames):
                path = self.get_color_filepath(scan_id, frame_id)
                if os.path.exists(path):
                    with Image.open(path) as image:
                        width, height = image.size
                    break
            else:
                raise FileNotFoundError(f"No images found for scan {scan_id}")
            self._intrinsics_cache[scan_id] = (K.astype(np.float32), height, width)
        return self._intrinsics_cache[scan_id]

    def _native_K_and_crop(self, scan_id, flip):
        """Native-resolution intrinsics after the (optional) training crop and
        flip, together with the crop box (left, top, right, bottom) or None."""
        K_33, height, width = self._load_native_intrinsics(scan_id)
        K = torch.eye(4, dtype=torch.float32)
        K[:3, :3] = torch.tensor(K_33)

        crop = None
        if self.split == "train":
            top, left, h, w = self.random_resize_crop.get_params(
                torch.empty((height, width)),
                self.random_resize_crop.scale,
                self.random_resize_crop.ratio,
            )
            K[0, 2] -= left
            K[1, 2] -= top
            width, height = w, h
            crop = (left, top, left + w, top + h)

        if flip:
            K[0, 2] = float(width) - K[0, 2]

        return K, crop, height, width

    def load_intrinsics(self, scan_id, frame_id=None, flip=False):
        """Loads intrinsics, computes scaled intrinsics, and returns a dict
        with intrinsics matrices for a frame at multiple scales.

        Returns:
            output_dict: A dict with
                - K_s{i}_b44 (intrinsics) and invK_s{i}_b44
                (backprojection) where i in [0,1,2,3,4]. i=0 provides
                intrinsics at the scale for depth_b1hw.
                - K_matching_b44 and invK_matching_b44 at the cost volume
                resolution.
                - K_full_depth_b44 and invK_full_depth_b44 provides
                intrinsics for the maximum available depth resolution.
                Only provided when include_full_depth_K is true.
            crop: (left, top, right, bottom) of the training crop or None.
        """
        K, crop, height, width = self._native_K_and_crop(scan_id, flip)
        # remembered so that load_lidar_prompt uses the same crop.
        self._last_native = (scan_id, int(frame_id), K.clone(), crop, flip)

        output_dict = {}

        if self.include_full_depth_K:
            output_dict["K_full_depth_b44"] = K.clone()
            output_dict["invK_full_depth_b44"] = torch.linalg.inv(K)

        K_matching = K.clone()
        K_matching[0] *= self.matching_width / float(width)
        K_matching[1] *= self.matching_height / float(height)
        output_dict["K_matching_b44"] = K_matching
        output_dict["invK_matching_b44"] = torch.linalg.inv(K_matching)

        K_depth = K.clone()
        K_depth[0] *= self.depth_width / float(width)
        K_depth[1] *= self.depth_height / float(height)

        for i in range(self.prediction_num_scales):
            K_scaled = K_depth.clone()
            K_scaled[:2] /= 2**i
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = torch.linalg.inv(K_scaled)

        return output_dict, crop

    def load_pose(self, scan_id, frame_id):
        """Loads a frame's pose.

        Returns:
            world_T_cam (numpy array): matrix for transforming from the
                camera to the world (pose).
            cam_T_world (numpy array): matrix for transforming from the
                world to the camera (extrinsics).
        """
        world_T_cam = np.eye(4, dtype=np.float64)
        world_T_cam[:3] = self._load_poses(scan_id)[int(frame_id)]
        cam_T_world = np.linalg.inv(world_T_cam)
        return world_T_cam.astype(np.float32), cam_T_world.astype(np.float32)

    # ------------------------------------------------------------------ #
    # depth
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_kitti_depth_png(path):
        """KITTI stores depth as uint16 PNG, metres * 256, 0 = invalid."""
        depth_png = np.array(Image.open(path), dtype=np.int32)
        assert depth_png.max() > 255 or depth_png.max() == 0, f"Not a 16 bit depth map: {path}"
        return depth_png.astype(np.float32) / 256.0

    def load_target_size_depth_and_mask(self, scan_id, frame_id, crop=None):
        """Loads a depth map at the resolution the dataset is configured for.

        NOTE: This function will place NaNs where depth maps are invalid.

        Returns:
            depth: depth map at the right resolution. Will contain NaNs
                where depth values are invalid.
            mask: a float validity mask for the depth maps. (1.0 where depth
            is valid).
            mask_b: like mask but boolean.
            skymask: (only if a sky mask exists on disk) 1 where sky.
        """
        depth = self._load_kitti_depth_png(self.get_full_res_depth_filepath(scan_id, frame_id))

        sky_path = self.get_sky_mask_filepath(scan_id, frame_id)
        sky = None
        if os.path.exists(sky_path):
            sky = (np.asarray(Image.open(sky_path).convert("L")) > 127).astype(np.uint8)

        if crop:
            depth = depth[crop[1] : crop[3], crop[0] : crop[2]]
            if sky is not None:
                sky = sky[crop[1] : crop[3], crop[0] : crop[2]]

        depth = cv2.resize(
            depth, dsize=(self.depth_width, self.depth_height), interpolation=cv2.INTER_NEAREST
        )
        depth = torch.tensor(depth).float().unsqueeze(0)
        mask_b = (depth > self.min_valid_depth) & (depth < self.max_valid_depth)
        mask = mask_b.float()
        depth[~mask_b] = torch.nan

        if sky is None:
            return depth, mask, mask_b

        sky = cv2.resize(
            sky, dsize=(self.depth_width, self.depth_height), interpolation=cv2.INTER_NEAREST
        )
        return depth, mask, mask_b, torch.tensor(sky).unsqueeze(0)

    def load_full_res_depth_and_mask(self, scan_id, frame_id):
        """Loads a depth map at the native resolution the dataset provides.

        NOTE: This function will place NaNs where depth maps are invalid.
        """
        full_res_depth = self._load_kitti_depth_png(
            self.get_full_res_depth_filepath(scan_id, frame_id)
        )
        full_res_depth = torch.tensor(full_res_depth).float().unsqueeze(0)
        full_res_mask_b = (full_res_depth > self.min_valid_depth) & (
            full_res_depth < self.max_valid_depth
        )
        full_res_mask = full_res_mask_b.float()
        full_res_depth[~full_res_mask_b] = torch.nan

        return full_res_depth, full_res_mask, full_res_mask_b

    # ------------------------------------------------------------------ #
    # prompts
    # ------------------------------------------------------------------ #
    def _load_native_lidar_prompt(self, scan_id, frame_id, K, crop, flip):
        """Raw LiDAR projection at native resolution, cropped / flipped like
        the image and thinned to the configured number of scan lines."""
        path = self.get_full_res_prompt_filepath(scan_id, frame_id)
        if not os.path.exists(path):
            # the depth benchmark has no projections for the first and last
            # five frames of a drive; such frames simply carry no prompt.
            _, height, width = self._load_native_intrinsics(scan_id)
            prompt = torch.zeros((1, height, width), dtype=torch.float32)
        else:
            prompt = torch.tensor(self._load_kitti_depth_png(path)).unsqueeze(0)
        if crop:
            prompt = prompt[:, crop[1] : crop[3], crop[0] : crop[2]]
        if flip:
            prompt = torch.flip(prompt, (-1,))
        return sample_lidar_lines(
            prompt,
            K,
            num_bins=self.lidar_num_bins,
            keep_every=self.lidar_keep_every,
            pitch_range=self.lidar_pitch_range,
        )

    def load_lidar_prompt(self, scan_id, frame_id, intrinsics, crop, flip):
        """Loads the LiDAR prompt of a frame at the depth resolution."""
        last = getattr(self, "_last_native", None)
        assert last is not None and last[0] == scan_id and last[1] == int(frame_id)
        K = last[2]
        prompt = self._load_native_lidar_prompt(scan_id, frame_id, K, crop, flip)
        return resize_sparse_depth(prompt, (self.depth_height, self.depth_width))

    def load_full_res_lidar_prompt(self, scan_id, frame_id, flip):
        """The prompt at native resolution, used to tell apart ground-truth
        pixels that coincide with a prompt sample from those that do not."""
        last = getattr(self, "_last_native", None)
        assert last is not None and last[0] == scan_id and last[1] == int(frame_id)
        if last[3] is not None:
            # native resolution outputs are only meaningful without a crop.
            return None
        return self._load_native_lidar_prompt(scan_id, frame_id, last[2], None, flip)
