from pathlib import Path

INPUT_ROOT = Path("shareleaks_results/results_total")

# Files to delete
FILES_TO_DELETE = {
    "S1_frame_metadata.csv",
    "S1_video_info.json",
    "S2_detections.json",
    "S5_key_candidates_cleaned.json",
    "S5_log.txt",
}


def delete_files():
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Directory not found: {INPUT_ROOT}")

    total_deleted = 0

    # Traverse all labels and users
    for label_dir in sorted(INPUT_ROOT.iterdir()):
        if not label_dir.is_dir():
            continue

        for user_dir in sorted(label_dir.iterdir()):
            if not user_dir.is_dir():
                continue

            deleted = 0

            for filename in FILES_TO_DELETE:
                file_path = user_dir / filename

                if file_path.is_file():
                    file_path.unlink()
                    deleted += 1
                    total_deleted += 1

            print(f"{label_dir.name}/{user_dir.name}: {deleted} files deleted")

    print("\n========== Summary ==========")
    print(f"Input:         {INPUT_ROOT}")
    print(f"Total deleted: {total_deleted}")
    print("=============================")


if __name__ == "__main__":
    delete_files()