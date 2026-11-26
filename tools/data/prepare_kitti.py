"""Writes camera poses and intrinsics for KITTI raw drives.

DriveMVS reads KITTI through ``mvsanywhere.datasets.kitti.KITTIDataset``,
which needs a pose for every frame and the intrinsics of the rectified
camera. Both are derived here from the official raw data:

    * intrinsics: ``P_rect_02`` of ``calib_cam_to_cam.txt``,
    * poses: the GPS/IMU (OXTS) trajectory, chained with the IMU -> Velodyne
      -> camera calibration to obtain camera-to-world transforms of the
      rectified camera. The world frame is the camera at the first frame.

Output, one folder per drive and camera:

    <kitti_root>/drivemvs_meta/<drive>/image_02/poses.txt        N x 12, rows of [R|t] (camera to world)
    <kitti_root>/drivemvs_meta/<drive>/image_02/intrinsics.txt   3 x 3

Usage:
    python tools/data/prepare_kitti.py --kitti_root data/kitti \
        --scans data_splits/drivemvs/kitti/val_scans.txt

DriveMVS addition; not part of the upstream MVSAnywhere code base.
"""

import argparse
from pathlib import Path

import numpy as np

EARTH_RADIUS = 6378137.0


def read_calib_file(path):
    """Reads a KITTI calibration file into a dict of float arrays."""
    data = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                data[key.strip()] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass  # e.g. calib_time
    return data


def rigid(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation).reshape(3)
    return transform


def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def oxts_to_imu_poses(oxts_files):
    """IMU-to-world poses from OXTS packets (Mercator projection, as in the
    KITTI raw data development kit)."""
    poses, scale = [], None
    for path in oxts_files:
        values = np.loadtxt(path)
        lat, lon, alt, roll, pitch, yaw = values[:6]
        if scale is None:
            scale = np.cos(lat * np.pi / 180.0)
        tx = scale * lon * np.pi * EARTH_RADIUS / 180.0
        ty = scale * EARTH_RADIUS * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
        rotation = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
        poses.append(rigid(rotation, [tx, ty, alt]))
    return poses


def camera_calibration(date_dir, camera):
    """Returns K (3x3) of the rectified camera and the IMU -> rectified
    camera transform."""
    cam = camera.split("_")[-1]  # '02'
    cam_to_cam = read_calib_file(date_dir / "calib_cam_to_cam.txt")
    velo_to_cam = read_calib_file(date_dir / "calib_velo_to_cam.txt")
    imu_to_velo = read_calib_file(date_dir / "calib_imu_to_velo.txt")

    projection = cam_to_cam[f"P_rect_{cam}"].reshape(3, 4)
    K = projection[:3, :3]

    rect = np.eye(4)
    rect[:3, :3] = cam_to_cam["R_rect_00"].reshape(3, 3)
    # offset of this camera from the rectified reference camera 0
    cam_from_cam0 = np.eye(4)
    cam_from_cam0[0, 3] = projection[0, 3] / projection[0, 0]

    cam_from_velo = cam_from_cam0 @ rect @ rigid(velo_to_cam["R"], velo_to_cam["T"])
    cam_from_imu = cam_from_velo @ rigid(imu_to_velo["R"], imu_to_velo["T"])
    return K, cam_from_imu


def process_scan(kitti_root, scan):
    """scan: '<split>/<date>_drive_<id>_sync/image_02'."""
    _, drive, camera = scan.split("/")
    date = drive.split("_drive")[0]
    date_dir = kitti_root / "raw_data" / date

    K, cam_from_imu = camera_calibration(date_dir, camera)
    oxts_files = sorted((date_dir / drive / "oxts" / "data").glob("*.txt"))
    if not oxts_files:
        raise FileNotFoundError(f"No OXTS packets found for {drive}")

    world_from_imu = oxts_to_imu_poses(oxts_files)
    imu_from_cam = np.linalg.inv(cam_from_imu)
    world_from_cam = [pose @ imu_from_cam for pose in world_from_imu]
    # express everything in the first camera frame; keeps numbers small.
    first_inv = np.linalg.inv(world_from_cam[0])
    poses = np.stack([(first_inv @ pose)[:3].reshape(-1) for pose in world_from_cam])

    out_dir = kitti_root / "drivemvs_meta" / drive / camera
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "poses.txt", poses, fmt="%.9e")
    np.savetxt(out_dir / "intrinsics.txt", K, fmt="%.9e")

    steps = np.linalg.norm(np.diff(poses.reshape(-1, 3, 4)[:, :, 3], axis=0), axis=1)
    print(f"{scan}: {len(poses)} frames, median motion per frame {np.median(steps):.3f} m")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kitti_root", required=True, type=Path)
    parser.add_argument("--scans", required=True, help="text file with one scan id per line")
    args = parser.parse_args()

    with open(args.scans) as f:
        scans = [line.strip() for line in f if line.strip()]
    for scan in scans:
        process_scan(args.kitti_root, scan)


if __name__ == "__main__":
    main()
