"""Evaluates a model on clips: depth accuracy and temporal consistency under
different prompt conditions.

For every requested prompt mode the whole split is run once:

    both      prompts for the reference frame and the source frames
    cur_only  prompt for the reference frame only
    src_only  prompts for the source frames only (reference frame is a blind spot)
    none      no prompts at all

Reported per mode (mean over frames, as in MVSAnywhere / DriveMVS):
    mae        mean absolute error in metres
    abs_rel    mean absolute relative error (%)
    a25        share of pixels with max(pred/gt, gt/pred) < 1.25 (%)
    *_np       the same on ground-truth pixels that are not within 2 px of a
               prompt sample of the reference frame ("no prompt" pixels)
    tae        temporal alignment error (%) between consecutive frames, see
               mvsanywhere/utils/temporal_metrics.py; tae_tol ignores pixel
               pairs that differ by more than 25 % (occlusions, moving objects)

The depth range of the cost volume never comes from the ground truth unless
--depth_range_mode gt is passed explicitly.

Examples:
    # DriveMVS checkpoint
    python tools/evaluate.py --config_file configs/drivemvs/drivemvs_vitb.yaml \
        --data_config_file configs/drivemvs/kitti_val.yaml \
        --load_weights_from_checkpoint path/to/drivemvs.ckpt \
        --eval_prompt_mode all --eval_output_json results/kitti.json

    # MVSAnywhere baseline through the same pipeline
    python tools/evaluate.py --config_file configs/drivemvs/mvsanywhere_baseline.yaml \
        --data_config_file configs/drivemvs/kitti_val.yaml \
        --lazy_load_weights_from_checkpoint weights/mvsanywhere_hero.ckpt \
        --eval_refinement_steps 1 --eval_output_json results/kitti_mvsa.json

DriveMVS addition; not part of the upstream MVSAnywhere code base.
"""

import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

import mvsanywhere.options as options
from mvsanywhere.train import prepare_model
from mvsanywhere.utils.dataset_utils import get_dataset
from mvsanywhere.utils.prompt_utils import PROMPT_MODES, apply_prompt_mode
from mvsanywhere.utils.temporal_metrics import temporal_alignment_error


def frame_metrics(gt, pred, valid):
    """Depth metrics of one frame on the pixels in `valid`."""
    if not valid.any():
        return None
    gt, pred = gt[valid].double(), pred[valid].double()
    ratio = torch.maximum(gt / pred, pred / gt)
    return {
        "mae": (gt - pred).abs().mean().item(),
        "abs_rel": 100 * ((gt - pred).abs() / gt).mean().item(),
        "a25": 100 * (ratio < 1.25).double().mean().item(),
    }


def build_model(opts):
    """DriveMVS checkpoints restore their own options; a few evaluation
    settings are taken from the command line."""
    if opts.load_weights_from_checkpoint is not None:
        from mvsanywhere.utils.model_utils import get_model_class

        # the checkpoint stores an Options object, which torch refuses to
        # unpickle with weights_only=True; only load checkpoints you trust.
        checkpoint = torch.load(opts.load_weights_from_checkpoint, map_location="cpu", weights_only=False)
        model_opts = checkpoint["hyper_parameters"]["opts"]
        model_opts.temporal_weights_path = None  # weights come from the checkpoint
        model_opts.da_weights_path = None
        model = get_model_class(model_opts)(model_opts)
        model.load_state_dict(checkpoint["state_dict"])
        for name in (
            "depth_range_mode",
            "disable_prompt_transfer",
            "fallback_min_depth",
            "fallback_max_depth",
            "prompt_range_near_scale",
            "prompt_range_far_scale",
        ):
            setattr(model.run_opts, name, getattr(opts, name))
    else:
        model = prepare_model(opts)
    return model.cuda().eval()


def main(opts):
    assert opts.datasets is not None, "Pass the dataset with --data_config_file."
    data_opts = opts.datasets[0]
    dataset_class, _ = get_dataset(data_opts.dataset, data_opts.dataset_scan_split_file, verbose=False)
    dataset = dataset_class(
        data_opts.dataset_path,
        split=data_opts.split,
        mv_tuple_file_suffix=data_opts.mv_tuple_file_suffix,
        tuple_info_file_location=data_opts.tuple_info_file_location,
        num_images_in_tuple=opts.num_images_in_tuple,
        image_width=opts.image_width,
        image_height=opts.image_height,
        include_full_res_depth=True,
        include_full_depth_K=True,
        pass_frame_id=True,
        disable_flip=True,
        matching_scale=opts.matching_scale,
        prediction_scale=opts.prediction_scale,
        prediction_num_scales=opts.prediction_num_scales,
        verbose_init=False,
    )
    dataset.configure_prompts(
        eval_num_lines=opts.prompt_eval_num_lines,
        **({"lidar_keep_every": opts.lidar_keep_every} if hasattr(dataset, "lidar_keep_every") else {}),
    )
    if opts.eval_max_clips is not None:
        stride = max(1, len(dataset) // opts.eval_max_clips)
        dataset.frame_tuples = dataset.frame_tuples[::stride][: opts.eval_max_clips]
    scan_of_clip = [dataset.parse_tuple(line)[0] for line in dataset.frame_tuples]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=opts.num_workers, drop_last=False
    )
    model = build_model(opts)

    modes = PROMPT_MODES if opts.eval_prompt_mode in ("all", "cycle") else (opts.eval_prompt_mode,)
    if not model.use_prompts:
        modes = ("none",)

    sums = {mode: defaultdict(float) for mode in modes}
    counts = {mode: defaultdict(int) for mode in modes}
    per_scan = {mode: defaultdict(lambda: defaultdict(list)) for mode in modes}
    previous = {mode: None for mode in modes}  # last frame of the previous clip

    def add(mode, scan, name, value):
        sums[mode][name] += value
        counts[mode][name] += 1
        per_scan[mode][scan][name].append(value)

    for clip_index, (cur_cpu, src_cpu) in enumerate(tqdm(loader, mininterval=30)):
        scan = scan_of_clip[clip_index]
        num_frames = cur_cpu["image_b3hw"].shape[1]
        for mode in modes:
            cur = {k: v.flatten(0, 1).cuda() for k, v in cur_cpu.items() if isinstance(v, torch.Tensor)}
            src = {k: v.flatten(0, 1).cuda() for k, v in src_cpu.items() if isinstance(v, torch.Tensor)}
            if "sparse_prompt_b1hw" in cur:
                apply_prompt_mode(cur, src, mode)

            use_gt_range = opts.depth_range_mode == "gt"
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=opts.precision == 16):
                outputs = model(
                    "test",
                    cur,
                    src,
                    unbatched_matching_encoder_forward=True,
                    num_refinement_steps=opts.eval_refinement_steps,
                    estimate_depth_range=not use_gt_range,
                    num_frames=num_frames,
                )
            pred = outputs["depth_pred_s0_b1hw"].float()

            # accuracy at the native ground-truth resolution
            gt = cur["full_res_depth_b1hw"]
            pred_full = F.interpolate(pred, size=gt.shape[-2:], mode="nearest")
            valid = torch.isfinite(gt) & (gt > 0.5)
            no_prompt = valid
            if "full_res_prompt_b1hw" in cur and mode in ("both", "cur_only"):
                near_prompt = F.max_pool2d((cur["full_res_prompt_b1hw"] > 0).float(), 5, 1, 2) > 0
                no_prompt = valid & ~near_prompt
            for i in range(num_frames):
                metrics = frame_metrics(gt[i], pred_full[i], valid[i])
                if metrics is None:
                    continue
                for name, value in metrics.items():
                    add(mode, scan, name, value)
                metrics_np = frame_metrics(gt[i], pred_full[i], no_prompt[i])
                if metrics_np is not None:
                    for name, value in metrics_np.items():
                        add(mode, scan, name + "_np", value)

            # temporal consistency between consecutive frames, also across
            # the border to the previous clip of the same scan.
            frames = [
                (pred[i, 0], cur["K_s0_b44"][i, :3, :3], cur["cam_T_world_b44"][i].double(), cur["world_T_cam_b44"][i].double())
                for i in range(num_frames)
            ]
            chain = frames
            if previous[mode] is not None and previous[mode][0] == scan and opts.eval_max_clips is None:
                chain = [previous[mode][1]] + frames
            for (d_a, K_a, _, c2w_a), (d_b, _, w2c_b, _) in zip(chain[:-1], chain[1:]):
                T = w2c_b @ c2w_a
                for name, tolerance in (("tae", None), ("tae_tol", 0.25)):
                    value = temporal_alignment_error(d_a, d_b, K_a, T, occlusion_tolerance=tolerance)
                    if value is not None:
                        add(mode, scan, name, 100 * value.item())
            previous[mode] = (scan, frames[-1])

    results = {"checkpoint": opts.load_weights_from_checkpoint or opts.lazy_load_weights_from_checkpoint,
               "dataset": data_opts.dataset, "split": data_opts.split, "num_clips": len(dataset),
               "depth_range_mode": opts.depth_range_mode, "refinement_steps": opts.eval_refinement_steps,
               "lidar_keep_every": getattr(dataset, "lidar_keep_every", None), "modes": {}, "per_scan": {}}
    names = ["mae", "abs_rel", "a25", "mae_np", "abs_rel_np", "a25_np", "tae", "tae_tol"]
    print("\n" + f"{'mode':10s}" + "".join(f"{n:>11s}" for n in names) + f"{'frames':>9s}")
    for mode in modes:
        results["modes"][mode] = {n: sums[mode][n] / max(1, counts[mode][n]) for n in names if counts[mode][n]}
        results["modes"][mode]["frames"] = counts[mode]["mae"]
        results["per_scan"][mode] = {
            scan: {n: sum(v) / len(v) for n, v in values.items()} for scan, values in per_scan[mode].items()
        }
        print(f"{mode:10s}" + "".join(f"{results['modes'][mode].get(n, float('nan')):11.3f}" for n in names) + f"{counts[mode]['mae']:9d}")

    if opts.eval_output_json:
        os.makedirs(os.path.dirname(os.path.abspath(opts.eval_output_json)), exist_ok=True)
        with open(opts.eval_output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {opts.eval_output_json}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    main(option_handler.options)
