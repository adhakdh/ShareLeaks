# -*- coding: utf-8 -*-
"""S3: planar test-grid geometry (labeled cells), no PIN interpretation.

Input: S2_detections.json (unchanged).
Output: S3_key_candidates.json (blue frames only), S3_mapping_debug/ (paired images).
Reads arbitrary cell labels from --layout, no label names are hard-coded.
"""
from pathlib import Path
import cv2
import numpy as np
from shareleaks_lib import (video_directories, read_json, write_json, read_image,
                            save_image, reset_step_directory, s3_homography,
                            s3_rank_candidates, s3_debug_pair)


def main(output='shareleaks_results', layout_path='S3_store_pin_layout.json', debug=True):
    layout=read_json(layout_path)
    if not layout.get('calibrated') or not layout.get('reference_corners') or not layout.get('keys'):
        raise ValueError('Provide a calibrated reference layout with corners and labeled rectangles')
    for directory in video_directories(output):
        records=[]
        debug_dir=reset_step_directory(directory,'S3_mapping_debug') if debug else None
        frames=read_json(directory/'S2_detections.json')['frames']
        for detection in frames:
            if not detection.get('blue_detected'): continue
            info={k:detection[k] for k in ('frame_id','source_frame_id','timestamp_ms','frame_name')}
            info.update(status='',rays=[])
            panel=detection.get('keyboard',{}).get('selected')
            rays=detection.get('ray',{}).get('candidates',[])
            if panel is None:
                info['status']='keyboard_missing'
            elif not rays:
                info['status']='ray_missing'
            else:
                try:
                    H=s3_homography(panel['corners'],layout)
                    for index,ray in enumerate(rays):
                        result=s3_rank_candidates(ray,H,layout)
                        result['ray_index']=index
                        info['rays'].append(result)
                    info['status']='multiple_rays' if len(rays)>1 else info['rays'][0]['status']
                    if debug:
                        frame=read_image(directory/'S1_frames'/detection['frame_name'])
                        # Debug chosen ray is longest, S2's selected_index when present.
                        index=detection['ray'].get('selected_index')
                        if not isinstance(index,int) or not 0<=index<len(rays): index=0
                        combined=s3_debug_pair(frame,np.array(panel['corners'],float),rays[index],layout,info['rays'][index])
                        save_image(debug_dir/detection['frame_name'],combined)
                except (ValueError,KeyError,np.linalg.LinAlgError,cv2.error) as ex:
                    info['status']='geometry_error'; info['error']=str(ex)
            records.append(info)
        write_json(directory/'S3_key_candidates.json',{
            'coordinate_system':'reference_photo_pixels',
            'distance_definition':'perpendicular center-to-infinite-projected-line',
            'threshold_rule':'distance <= min(cell_width, cell_height)/2',
            'frames':records})
        print(f'S3: {directory} | {len(records)} blue frames')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='shareleaks_results')
    parser.add_argument('--layout',default='S3_store_pin_layout.json')
    parser.add_argument('--no-debug',action='store_true')
    args=parser.parse_args()
    main(args.output,args.layout,not args.no_debug)
