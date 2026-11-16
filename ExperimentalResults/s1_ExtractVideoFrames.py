# -*- coding: utf-8 -*-

from pathlib import Path
from shareleaks_lib import discover_videos, split_one_video


def main(
    input_dir="dataset_raw_split",
    save_dir="shareleaks_results",
    selected_labels=None,
    interval_ms=100,
):
    videos = discover_videos(input_dir, selected_labels or None)

    if not videos:
        raise RuntimeError("xxxxx No videos found in input directory.")

    for item in videos:
        video_path = Path(item["video_path"])
        label = video_path.parent.stem
        video_dir = Path(save_dir) / label / video_path.stem

        split_one_video(
            video_path=video_path,
            label=label,
            video_dir=video_dir,
            interval_ms=interval_ms,
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='dataset_raw_split')
    parser.add_argument('--output', default='shareleaks_results')
    parser.add_argument('--labels', nargs='*', default=['4'],
                        help='Empty list means all labels')
    parser.add_argument('--interval-ms', type=float, default=100,
                        help='Time interval between extracted frames (ms)')

    args = parser.parse_args()

    if args.interval_ms <= 0:
        parser.error('--interval-ms must be > 0')

    main(args.input, args.output, args.labels, args.interval_ms)