# -*- coding: utf-8 -*-
"""S4：按蓝色状态分段，再对段内按键多数投票。"""

"""
候选集合投票（Candidate-set Voting）：每帧将满足阈值的候选标签视为一个整体，统计相同候选集合在事件中出现的次数。
距离最小代表帧（Minimum-distance Representative）：选出出现次数最多的候选集合后，在对应帧中选择 Top-1 几何距离最小的帧作为代表。

当前的s4只是整合事件，还需要一个步骤，那就是计算根据射线的位置，对候选项进行进一步筛选，比如第一个蓝色射线帧的命中按键是2和5，下一帧是6和7，需要计算射线的位置变化，来判断出射线是在往上还是往下，往左还是右边，进而对候选项进行推断，是吗？先整合事件，就是连续帧就是一次点击，可以选其中top1距离最小的那个作为当前事件的结果，然后再判断射线的位置，相对于上一个位置的变化，

"""

# 在进入s4之前，还需要一个步骤，那就是计算根据射线的位置，对候选项进行进一步筛选，比如第一个蓝色射线帧的命中按键是2和5，下一帧是6和7，那么


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
                events = [events[0],events[1],events[2],events[4]]# 只保留前5个时间，这个参数
            else:
                events = []
        for event in events:
            event['all_frame_names'] = ';'.join(event['all_frame_names'])
        write_csv(directory / 'S4_key_events.csv', events, FIELDS)
        clean_s3_mapping_debug(directory, events)   # 新增：清理 S3_mapping_debug，只保留 S4 选中的帧
        print(f'S4: {directory}: {len(events)} blue-ray events')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='shareleaks_results')
    parser.add_argument('--max-missing-frames', type=int, default=0)
    args = parser.parse_args()
    if args.max_missing_frames < 0:
        parser.error('--max-missing-frames must be >= 0')
    main(args.output, args.max_missing_frames)
