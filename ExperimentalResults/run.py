# -*- coding: utf-8 -*-
"""统一入口：修改 main() 内的参数后直接运行。"""

from pathlib import Path
import shutil
import s1_ExtractVideoFrames as s1
import s2_DetectRayKeyboard as s2
import s3_MapKeyboardKeys as s3
import s4_ExtractKeyEvents as s4
import s5_rebuilt as s5
import s6_SavedResults as s6
import s7_PrintResults as s7
import os


def run_main(tmp_str = "3"):
    input_dir = f"dataset_raw_split/split_{tmp_str}"
    save_dir = f"shareleaks_results/results_{tmp_str}"

    selected_labels = [tmp_str] 
    steps = [1,2,3,4,5,6] 
    steps = [5,6,7]
    clear_output_on_s1 =  True


    interval_ms = 200
    layout_path = "keyboard/storepin/storepin.json"
    if tmp_str in ["unlock4","unlock6",'unlock8']:
        layout_path = "keyboard/unlock/unlock.json"
    if tmp_str in ["keypad4","keypad6",'keypad8']:
        layout_path = "keyboard/keypad/keypad.json"
    save_debug_images = True 
    events_num = tmp_str 
    max_endpoint_extension = 0
    max_missing_frames = 0 
    ground_truth_path = NotImplementedError



    if max_endpoint_extension < 0 or max_missing_frames < 0:
        raise ValueError("xx")

    # ===== 按顺序执行选定步骤 =====
    if 1 in steps:
        if clear_output_on_s1:
            output_root = Path(save_dir)
            if output_root.exists():
                exit("xxxxx Output directory already exists. Please remove it or set clear_output_on_s1=False.")
                shutil.rmtree(output_root)
            output_root.mkdir(parents=True, exist_ok=True)
        s1.main(
            input_dir=input_dir,
            save_dir=save_dir,
            selected_labels=selected_labels,
            interval_ms=interval_ms,
        )

    if 2 in steps:
        s2.main(
            tmp_str = tmp_str,
            save_dir=save_dir,
            debug=save_debug_images,
        )

    if 3 in steps:
        s3.main(
            output=save_dir,
            layout_path=layout_path,
            debug=save_debug_images,
        )

    if 4 in steps:
        s4.main(tmp_str = tmp_str,
            save_dir=save_dir,
            max_missing_frames=max_missing_frames,
            events_num = events_num,
        )

    if 5 in steps:
        s5.main(
            output=save_dir
        )

    if 6 in steps:
        s6.main(output=save_dir)     
    if 7 in steps:
        s7.main(tmp_str=tmp_str,input_path=save_dir)    

if __name__ == "__main__":
    tmp_list = ['total'] 
    for tmp_str in tmp_list:
        run_main(tmp_str)