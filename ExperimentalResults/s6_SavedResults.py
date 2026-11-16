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
from shareleaks_lib import read_positions, evaluate_trial, save_trial_results 


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
