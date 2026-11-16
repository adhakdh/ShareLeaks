
# -*- coding: utf-8 -*-
"""Rename ShareLeaks user directories using participant groups.

Example:
    results_total/0133/hfl_5_2 -> results_total/0133/user1_1
    results_total/0133/hfl_5_1 -> results_total/0133/user1_2
    results_total/0133/hfl_7_2 -> results_total/0133/user2_1

Only directory names are changed.
Files inside directories remain unchanged.
"""

from pathlib import Path
import re


def get_participant_groups():
    return [
        ['5_2', '5_1', '6_1', '6_2', '7_1'],
        ['7_2', '8_1', '8_2', '9_1', '9_2'],
        ['10_1', '10_2', '11_1', '11_2', '1_1'],
        ['12_1', '12_2', '13_1', '13_2', '1_2'],
        ['15_1', '15_2', '16_1', '16_2', '25_1'],
        ['17_1', '17_2', '18_1', '18_2', '25_2'],
        ['19_1', '19_2', '20_1', '20_2', '2_1'],
        ['21_1', '21_2', '22_1', '22_2', '2_2'],
        ['23_1', '23_2', '24_1', '24_2', '3_1'],
        ['26_1', '26_2', '37_1', '37_2', '3_2'],
        ['27_1', '27_2', '28_1', '28_2', '4_1'],
        ['29_1', '29_2', '30_1', '30_2', '4_2'],
        ['31_1', '31_2', '32_1', '32_2', '14_1'],
        ['33_1', '33_2', '34_1', '34_2', '14_2'],
        ['35_1', '35_2', '36_1', '36_2', '50_1'],
        ['38_1', '38_2', '39_1', '39_2', '50_2'],
        ['40_1', '40_2', '41_1', '41_2', '55_1'],
        ['42_1', '42_2', '43_1', '43_2', '55_2'],
        ['44_1', '44_2', '45_1', '45_2', '54_1'],
        ['46_1', '46_2', '47_1', '47_2', '54_2'],
        ['48_1', '48_2', '49_1', '49_2', '53_1'],
        ['51_1', '51_2', '52_1', '52_2', '53_2'],
    ]


def build_rename_map(groups):
    """Build old suffix -> new directory name mapping."""

    rename_map = {}

    for user_index, codes in enumerate(groups, start=1):
        for trial_index, code in enumerate(codes, start=1):
            if code in rename_map:
                raise ValueError(f"Duplicate code: {code}")

            rename_map[code] = f"user{user_index}_{trial_index}"

    return rename_map


def extract_code(directory_name):
    """Extract trailing code, e.g. hfl_5_2 -> 5_2."""

    match = re.search(r'_(\d+_\d+)$', directory_name)

    if match:
        return match.group(1)

    return None


def rename_directories(input_root, rename_map, dry_run=False):
    """Rename user directories under every label."""

    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_root}"
        )

    rename_tasks = []
    skipped = []

    # Step 1: Collect rename operations
    for label_dir in sorted(input_root.iterdir()):
        if not label_dir.is_dir():
            continue

        for user_dir in sorted(label_dir.iterdir()):
            if not user_dir.is_dir():
                continue

            code = extract_code(user_dir.name)

            if code not in rename_map:
                skipped.append(user_dir)
                continue

            new_name = rename_map[code]
            new_path = label_dir / new_name

            if new_path == user_dir:
                continue

            rename_tasks.append((user_dir, new_path))

    # Step 2: Validate all destination paths
    sources = {src for src, _ in rename_tasks}
    destinations = [dst for _, dst in rename_tasks]

    if len(destinations) != len(set(destinations)):
        raise ValueError(
            "Duplicate destination directories detected."
        )

    for src, dst in rename_tasks:
        if dst.exists() and dst not in sources:
            raise FileExistsError(
                f"Destination already exists: {dst}"
            )

    # Avoid collisions with other directories being renamed.
    if any(dst in sources for dst in destinations):
        raise ValueError(
            "Destination overlaps an existing source directory. "
            "Please resolve naming conflicts before renaming."
        )

    # Step 3: Rename directories
    for src, dst in rename_tasks:
        print(f"{src.parent.name}/{src.name} -> {dst.name}")

        if not dry_run:
            src.rename(dst)

    print("\n========== Summary ==========")
    print(f"Input:        {input_root}")
    print(f"Renamed:      {len(rename_tasks)}")
    print(f"Skipped:      {len(skipped)}")
    print(f"Dry run:      {dry_run}")
    print("=============================")

    if skipped:
        print("\nSkipped directories:")
        for path in skipped:
            print(f"  {path}")


def main():
    input_root = Path("shareleaks_results/results_total")

    groups = get_participant_groups()
    rename_map = build_rename_map(groups)

    rename_directories(
        input_root=input_root,
        rename_map=rename_map,
        dry_run=False,
    )


if __name__ == "__main__":
    main()
