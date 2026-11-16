# -*- coding: utf-8 -*-
"""S2: filter detected rays by keyboard intersection, then select longest."""
import argparse
import math
from shareleaks_lib import (video_directories, read_csv, read_image, detect_blue_ray,
                           detect_keyboard, draw_detection, save_image, write_json, select_ray_by_keyboard)

def main(tmp_str = '', save_dir='shareleaks_results', debug=True):
    for directory in video_directories(save_dir):
        records = []
        for metadata in read_csv(directory/'S1_frame_metadata.csv'):
            frame = read_image(directory/'S1_frames'/metadata['frame_name'])
            ray = detect_blue_ray(frame)
            # print(ray)
            # exit()
            keyboard = detect_keyboard(tmp_str,frame) if ray['blue_detected'] else {'selected': None, 'candidates': []}
            # print(keyboard)
            # exit()
            ray = select_ray_by_keyboard(tmp_str,ray, keyboard)
            # print(ray,keyboard)
            # exit()
            row = dict(metadata)
            row.update(frame_id=int(row['frame_id']), source_frame_id=int(row['source_frame_id']),
                       timestamp_ms=float(row['timestamp_ms']), blue_detected=ray['blue_detected'],
                       ray=ray, keyboard=keyboard)
            records.append(row)
            if debug and ray['blue_detected']:
                save_image(directory/'S2_detections_debug'/row['frame_name'],
                           draw_detection(frame, ray, keyboard))
        write_json(directory/'S2_detections.json', {'schema_version': 1, 'frames': records})
        print(f'S2: {directory}: {sum(r["blue_detected"] for r in records)}/{len(records)} valid ray frames')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='shareleaks_results')
    parser.add_argument('--no-debug', action='store_true')
    args = parser.parse_args()
    main(args.output, not args.no_debug)
