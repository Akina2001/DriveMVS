"""Groups per-frame MVS tuples into temporally consecutive clips.

DriveMVS trains and evaluates on short clips so that the temporal decoder sees
neighbouring frames. A per-frame tuple file has one line per reference frame:

    scan_id ref src src ...

A clip file has one line per clip, with the tuples of ``--clip_length``
consecutive reference frames of the same scan separated by ';':

    scan_id;ref_0 src src ...;ref_1 src src ...;...

Example:
    python tools/data/make_clip_tuples.py \
        --input data_splits/drivemvs/kitti/val_eight_view_deepvmvs.txt \
        --output data_splits/drivemvs/kitti/val_eight_view_deepvmvs_clip6.txt \
        --clip_length 6

DriveMVS addition; not part of the upstream MVSAnywhere code base.
"""

import argparse
import random
import re
from collections import defaultdict


def frame_number(frame_id: str) -> int:
    """'000645_left' -> 645, '0066' -> 66."""
    match = re.match(r"(\d+)", frame_id)
    if match is None:
        raise ValueError(f"Cannot read a frame number from '{frame_id}'")
    return int(match.group(1))


def frame_stream(frame_id: str) -> str:
    """Part of the frame id that is not the number, e.g. '_left'. Frames of
    different streams never share a clip."""
    return re.sub(r"^\d+", "", frame_id)


def make_clips(lines, clip_length, stride, max_gap, keep_remainder=False):
    """Returns clip lines built from per-frame tuple lines."""
    per_scan = defaultdict(dict)
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        scan_id, ref = parts[0], parts[1]
        # keep the first tuple if a reference frame is listed more than once.
        per_scan[(scan_id, frame_stream(ref))].setdefault(frame_number(ref), " ".join(parts[1:]))

    clips = []
    for (scan_id, _), tuples in sorted(per_scan.items()):
        numbers = sorted(tuples)
        # split the scan into runs without temporal gaps
        runs, run = [], [numbers[0]]
        for number in numbers[1:]:
            if number - run[-1] <= max_gap:
                run.append(number)
            else:
                runs.append(run)
                run = [number]
        runs.append(run)

        for run in runs:
            starts = list(range(0, len(run) - clip_length + 1, stride))
            for start in starts:
                clip = run[start : start + clip_length]
                clips.append(scan_id + ";" + ";".join(tuples[n] for n in clip))
            if keep_remainder:
                covered = starts[-1] + clip_length if starts else 0
                if covered < len(run):
                    clip = run[covered:]
                    clips.append(scan_id + ";" + ";".join(tuples[n] for n in clip))
    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help="per-frame tuple file")
    parser.add_argument("--output", required=True, help="clip tuple file to write")
    parser.add_argument("--clip_length", type=int, default=6)
    parser.add_argument("--stride", type=int, default=None, help="default: clip_length")
    parser.add_argument("--max_gap", type=int, default=1,
                        help="largest frame-number step allowed inside a clip")
    parser.add_argument("--keep_remainder", action="store_true",
                        help="also write the shorter clip left at the end of a scan")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the clips")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        lines = [line.strip() for line in f if line.strip()]

    clips = make_clips(
        lines, args.clip_length, args.stride or args.clip_length, args.max_gap, args.keep_remainder
    )
    if args.shuffle:
        random.Random(args.seed).shuffle(clips)

    with open(args.output, "w") as f:
        f.write("\n".join(clips) + "\n")
    print(f"{len(lines)} tuples -> {len(clips)} clips of length {args.clip_length}: {args.output}")


if __name__ == "__main__":
    main()
