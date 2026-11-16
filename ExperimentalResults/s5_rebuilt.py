
# -*- coding: utf-8 -*-
"""
S5: Event-level key ranking based on RMS Distance.

Inputs:
    S3_key_candidates.json
    S4_key_events.csv

Outputs:
    S5_key_candidates_cleaned.json
    S5_log.txt

Method:
    1. Read all blue frames of each event from S4.
    2. Extract candidate distances from S3.
    3. Calculate RMS Distance for every candidate key.
    4. Rank all candidates by RMS Distance.
    5. Mark the best candidate with hit=True.

    RMS = sqrt(sum(distance^2) / N)

    No original hit-based filtering is applied.
    No candidate distance is forced to zero.

Usage:
    python s6_rebuilt.py --output shareleaks_results
"""

import argparse
import csv
import json
import math
from pathlib import Path


def read_json(path):
    with path.open('r', encoding='utf-8-sig') as f:
        return json.load(f)


def read_s4(path):
    """Read S4 events and their associated blue frames."""
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError('S4 contains no events')

    for row in rows:
        if not row.get('frame_name'):
            raise ValueError('S4 missing frame_name')

        if not row.get('all_frame_names'):
            raise ValueError('S4 missing all_frame_names')

        row['all_frame_names'] = [
            name.strip()
            for name in row['all_frame_names'].split(';')
            if name.strip()
        ]

        if row['frame_name'] not in row['all_frame_names']:
            raise ValueError(
                f"{row['frame_name']}: representative frame "
                f"not found in all_frame_names"
            )

    return rows


def get_ranking(frame):
    """
    Extract candidate ranking from the first ray.

    Consistent with the original S5 implementation.
    """
    rays = frame.get('rays') or []

    if not rays:
        raise ValueError(
            f"{frame['frame_name']}: rays is empty"
        )

    ranking = rays[0].get('ranking', [])

    if not ranking:
        raise ValueError(
            f"{frame['frame_name']}: ranking is empty"
        )

    return ranking


def get_distances(frame):
    """Extract key-distance mapping from the first ray."""
    ranking = get_ranking(frame)

    distances = {}

    for item in ranking:
        key = str(item['key'])
        distance = float(item['distance'])

        if not math.isfinite(distance) or distance < 0:
            raise ValueError(
                f"{frame['frame_name']}: invalid distance for {key}"
            )

        if key in distances:
            raise ValueError(
                f"{frame['frame_name']}: duplicate key {key}"
            )

        distances[key] = distance

    return distances


def calculate_score(key, event_distances):
    """
    Calculate RMS Distance for a candidate key.

        RMS = sqrt(mean(distance^2))

    Lower RMS indicates better geometric consistency
    with the event's rays.
    """
    distances = []

    for frame_name, frame_distances in event_distances:
        if key not in frame_distances:
            raise ValueError(
                f'{frame_name}: missing key {key}'
            )

        distances.append(frame_distances[key])

    return math.sqrt(
        sum(d ** 2 for d in distances) / len(distances)
    )


def filter_event(row, frames):
    """
    Process one event using RMS Distance.

    All candidate keys are evaluated across all blue frames.

    No original hit-based filtering.
    No forced zero distance.

    The best candidate is marked with hit=True.
    """
    representative_name = row['frame_name']
    event_names = row['all_frame_names']

    if representative_name not in frames:
        raise ValueError(
            f'Representative frame not found: {representative_name}'
        )

    # Collect distance mappings from all blue frames.
    event_distances = []

    for name in event_names:
        if name not in frames:
            raise ValueError(
                f'Event frame not found in S3: {name}'
            )

        distances = get_distances(frames[name])
        event_distances.append((name, distances))

    # Obtain all candidate keys from the first event frame.
    keys = list(event_distances[0][1].keys())

    # Ensure every frame contains the same candidate keys.
    expected_keys = set(keys)

    for name, distances in event_distances:
        if set(distances) != expected_keys:
            raise ValueError(
                f'{name}: inconsistent candidate keys'
            )

    # Calculate RMS Distance for every candidate.
    scores = {}

    for key in keys:
        scores[key] = calculate_score(
            key,
            event_distances
        )

    # Select the candidate with the minimum RMS Distance.
    best_key = min(
        keys,
        key=lambda key: (
            scores[key],
            key
        )
    )

    # Construct final ranking.
    # Keep the original key/distance/hit structure.
    output_ranking = [
        {
            'key': key,
            'distance': scores[key],
            'hit': key == best_key
        }
        for key in keys
    ]

    output_ranking.sort(
        key=lambda item: (
            item['distance'],
            item['key']
        )
    )

    # Keep representative frame name for compatibility.
    # Its distance values are not used in scoring.
    output_frame = {
        'frame_name': representative_name,
        'ranking': output_ranking
    }

    # Detailed event log.
    log = [
        f"Event {row.get('event_id', '?')}",
        f"Representative: {representative_name}",
        f"Frames: {event_names}",
        f"Frame count: {len(event_names)}",
        "RMS ranking:"
    ]

    for rank, item in enumerate(output_ranking, start=1):
        log.append(
            f"  {rank:2d}. "
            f"Key {item['key']}: "
            f"RMS = {item['distance']:.6f}"
        )

    log.extend([
        f"Selected: {best_key}",
        ""
    ])

    return output_frame, '\n'.join(log)


def process(s3_path, s4_path, dest):
    """Process all events in one experiment directory."""
    s3 = read_json(s3_path)
    rows = read_s4(s4_path)

    frames = {
        frame['frame_name']: frame
        for frame in s3.get('frames', [])
    }

    output_frames = []
    logs = []

    for row in rows:
        output_frame, log = filter_event(row, frames)

        output_frames.append(output_frame)
        logs.append(log)

    result = {
        'frames': output_frames
    }

    # Write files only after all events have been processed.
    dest.mkdir(parents=True, exist_ok=True)

    output_path = dest / 'S5_key_candidates_cleaned.json'

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8'
    )

    log_path = dest / 'S5_log.txt'

    log_path.write_text(
        '\n'.join(logs),
        encoding='utf-8'
    )

    return len(output_frames)


def main(output='shareleaks_results'):
    root = Path(output)

    s4_paths = sorted(root.rglob('S4_key_events.csv'))

    if not s4_paths:
        print(f'No S4_key_events.csv found under {root}')
        return

    success = 0
    failed = 0

    for s4_path in s4_paths:
        directory = s4_path.parent

        try:
            count = process(
                directory / 'S3_key_candidates.json',
                s4_path,
                directory
            )

            print(
                f'S5: {directory} | OK | '
                f'{count} events processed'
            )

            success += 1

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError
        ) as exc:

            print(f'S5: {directory} | ERROR: {exc}')
            failed += 1

    print()
    print(f'Total experiments: {len(s4_paths)}')
    print(f'Success: {success}')
    print(f'Failed: {failed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--output',
        default='shareleaks_results'
    )

    args = parser.parse_args()

    main(args.output)
