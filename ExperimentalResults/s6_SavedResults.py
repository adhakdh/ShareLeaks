# -*- coding: utf-8 -*-
"""Character-level and string-level Top-K accuracy from S5 ranking data.
"""
import argparse
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
import csv

def save_trial_results(root, rows, char_ks, string_ks):
    """Save per-trial Top-K accuracies as percentages."""

    output_path = root / 'topk_per_trial.csv'

    fieldnames = ['Label', 'Experiment', 'Trial']
    fieldnames += [f'C@{k}' for k in char_ks]
    fieldnames += [f'S@{k}' for k in string_ks]

    with output_path.open(
        'w', newline='', encoding='utf-8-sig'
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n[CSV] Saved {len(rows)} trials to: {output_path}')

def read_positions(path):
    with path.open('r', encoding='utf-8-sig') as f:
        data = json.load(f)
    frames = data.get('frames', [])
    positions = []
    for frame in frames:
        ranking = frame.get('ranking', [])
        choices = []
        for original_index, entry in enumerate(ranking):
            distance = entry.get('distance')
            key = str(entry.get('key', ''))
            if (len(key) == 1 and key.isdigit() and
                    isinstance(distance, (int, float)) and
                    not isinstance(distance, bool) and math.isfinite(distance)):
                choices.append((key, float(distance), original_index))
        choices.sort(key=lambda item: item[1])
        if not choices:
            raise ValueError(f"frame {frame.get('frame_name')}: no valid digit choices")
        positions.append(choices)
    return positions


def evaluate_trial(positions, label, char_ks, string_ks):
    import heapq

    if len(positions) != len(label):
        raise ValueError(
            f'Position count ({len(positions)}) '
            f'does not match label length ({len(label)})'
        )

    char_ranks = []
    for choices, target in zip(positions, label):
        rank = next(
            (i + 1 for i, (key, _, _) in enumerate(choices)
             if key == target),
            math.inf
        )
        char_ranks.append(rank)

    # Generate only the Top-K combinations
    max_k = max(string_ks)
    initial = tuple(0 for _ in positions)

    initial_distance = sum(p[0][1] for p in positions)
    heap = [(initial_distance, initial)]
    visited = {initial}

    string_rank = math.inf

    for rank in range(1, max_k + 1):
        if not heap:
            break

        distance, indices = heapq.heappop(heap)

        word = ''.join(
            positions[i][j][0]
            for i, j in enumerate(indices)
        )

        if word == label:
            string_rank = rank
            break

        for i in range(len(positions)):
            if indices[i] + 1 >= len(positions[i]):
                continue

            next_indices = list(indices)
            next_indices[i] += 1
            next_indices = tuple(next_indices)

            if next_indices in visited:
                continue

            visited.add(next_indices)

            next_distance = (
                distance
                - positions[i][indices[i]][1]
                + positions[i][next_indices[i]][1]
            )

            heapq.heappush(
                heap,
                (next_distance, next_indices)
            )

    return (
        {k: sum(rank <= k for rank in char_ranks) for k in char_ks},
        {k: int(string_rank <= k) for k in string_ks}
    )


def original_stem(stem):
    DECIMAL_SUFFIX = re.compile(r'^\d+\.\d+$')
    """Strip consecutive trailing decimal metrics, e.g. _37.5_75.0."""
    parts = stem.split('_')
    while len(parts) > 1 and DECIMAL_SUFFIX.fullmatch(parts[-1]):
        parts.pop()
    return '_'.join(parts)

def main(output):
    char_ks=[1, 3]
    string_ks=[1, 3, 5, 10, 20, 30, 50, 100]
    filename_metrics= ['C@3', 'S@50']  # Change order or remove metrics as needed.
    FILENAME = 'S5_key_candidates_cleaned.json'
    
    # Ensure the requested filename metrics are computed even when absent
    # from the printed summary's Top-K settings.
    required_char_ks = set(char_ks)
    required_string_ks = set(string_ks)
    for metric in filename_metrics:
        match = re.fullmatch(r'([CS])@([1-9]\d*)', metric)
        if not match:
            raise ValueError(f'Invalid filename metric: {metric!r}')
        kind, k = match.group(1), int(match.group(2))
        (required_char_ks if kind == 'C' else required_string_ks).add(k)
    if not filename_metrics:
        raise ValueError('FILENAME_METRICS must contain at least one metric')

    root = Path(output)
    paths = sorted(root.rglob(FILENAME))
    if not paths:
        print(f'No {FILENAME} found under {root}')
        return
    stats = defaultdict(lambda: {'chars': 0, 'strings': 0,
                                 'char_hits': defaultdict(int),
                                 'string_hits': defaultdict(int)})
    skipped = 0
    trial_rows = []
    for path in paths:
        label = path.parent.parent.name
        label_length = len(label)
        try:
            positions = read_positions(path)
            if len(positions) % label_length != 0:
                raise ValueError(
                    f'{len(positions)} frames cannot be split into '
                    f'groups of {label_length}'
                )

            trials = [
                positions[i:i + label_length]
                for i in range(0, len(positions), label_length)
            ]

            if not trials:
                raise ValueError('no frames')
            outcomes = [evaluate_trial(trial, label, required_char_ks,
                                       required_string_ks) for trial in trials]

            for trial_index, (ch, st) in enumerate(outcomes, start=1):
                result = {'Label': label,'Experiment': path.parent.name,'Trial': trial_index,}
                for k in char_ks:
                    result[f'C@{k}'] = round(100 * ch[k] / label_length, 1)
                for k in string_ks:
                    result[f'S@{k}'] = round(100 * st[k], 1)
                trial_rows.append(result)

            row = stats[label]
            for ch, st in outcomes:
                row['chars'] += label_length
                row['strings'] += 1
                for k in char_ks:
                    row['char_hits'][k] += ch[k]
                for k in string_ks:
                    row['string_hits'][k] += st[k]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(f'[SKIP] {path}: {exc}')
            skipped += 1
            continue

    # columns = [('Label', 8), ('N', 5)]
    # columns += [(f'C@{k}', 7) for k in char_ks]
    # columns += [(f'S@{k}', 7) for k in string_ks]
    # header = ' '.join(f'{name:>{width}}' for name, width in columns)
    # print('\n' + header)
    # print('-' * len(header))

    # overall = {'chars': 0, 'strings': 0,
    #            'char_hits': defaultdict(int), 'string_hits': defaultdict(int)}

    # def print_row(label, row):
    #     values = [label, str(row['strings'])]
    #     values += [f"{100 * row['char_hits'][k] / row['chars']:.1f}" if row['chars'] else 'N/A'
    #                for k in char_ks]
    #     values += [f"{100 * row['string_hits'][k] / row['strings']:.1f}" if row['strings'] else 'N/A'
    #                for k in string_ks]
    #     print(' '.join(f'{v:>{width}}' for v, (_, width) in zip(values, columns)))

    # for label, row in sorted(stats.items()):
    #     print_row(label, row)
    #     overall['chars'] += row['chars']
    #     overall['strings'] += row['strings']
    #     for k in char_ks:
    #         overall['char_hits'][k] += row['char_hits'][k]
    #     for k in string_ks:
    #         overall['string_hits'][k] += row['string_hits'][k]
    # print('-' * len(header))
    # print_row('OVERALL', overall)
    # print(f'\nFiles found: {len(paths)}; skipped: {skipped}; evaluated strings: {len(paths) - skipped}')
    save_trial_results(root, trial_rows, char_ks, string_ks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='shareleaks_results')
    args = parser.parse_args()
    main(args.output)
