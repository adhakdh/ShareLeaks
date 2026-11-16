# -*- coding: utf-8 -*-
import argparse
from shareleaks_lib import video_directories, read_json, write_csv, write_json, extract_blue_events, clean_s3_mapping_debug


from pathlib import Path


def main(tmp_str='',save_dir='shareleaks_results', max_missing_frames=0, events_num = 5):
    FIELDS = [
    'event_id', 'frame_name','all_frame_names','start_ms', 'end_ms', 'observed_span_ms',
    'sampling_interval_ms', 'blue_frame_count', 'first_frame_id',
    'last_frame_id', 'representative_frame_id', 'status'
]
    for directory in video_directories(save_dir):
        info = read_json(directory / 'S1_video_info.json')
        detections = read_json(directory / 'S2_detections.json')['frames']
        events = extract_blue_events(detections, info, max_missing_frames)

        if tmp_str in ['unlock4','keypad4']:
            events = events[0:4] if (len(events) >= 4) else []
        elif tmp_str in ['unlock6','keypad6']:
            events = events[0:6] if (len(events) >= 6) else []
        elif tmp_str in ['unlock8','keypad8']:
            events = events[0:8] if (len(events) >= 8) else []
        else:
            if (len(events) >= 5):
                events = [events[0],events[1],events[2],events[4]]
            else:
                events = []
        for event in events:
            event['all_frame_names'] = ';'.join(event['all_frame_names'])
        write_csv(directory / 'S4_key_events.csv', events, FIELDS)
        clean_s3_mapping_debug(directory, events) 
        print(f'S4: {directory}: {len(events)} blue-ray events')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='shareleaks_results')
    parser.add_argument('--max-missing-frames', type=int, default=0)
    args = parser.parse_args()
    if args.max_missing_frames < 0:
        parser.error('--max-missing-frames must be >= 0')
    main(args.output, args.max_missing_frames)
