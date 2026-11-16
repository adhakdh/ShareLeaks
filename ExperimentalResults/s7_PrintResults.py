# -*- coding: utf-8 -*-
"""Compact ShareLeaks Top-K summary: one row per participant and two overall rows.

Usage:
    python analyze_topk_compact.py --input shareleaks_results/results_right_eye
    python analyze_topk_compact.py --input shareleaks_results/results_v_1s/topk_per_trial.csv

Cells use percentage units. The macro row reports mean ± sample SD across participants.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

# Configure groups: codes within a nested list belong to the same participant.

METRICS = ['C@1', 'C@3', 'S@1', 'S@3', 'S@5', 'S@10',
           'S@20', 'S@30', 'S@50', 'S@100']

def build_code_map(groups):
    result = {}
    for person_index, codes in enumerate(groups, start=1):
        if not codes:
            raise ValueError(f'Participant P{person_index:02d} has no codes')
        for code in codes:
            code = str(code)
            if code in result:
                raise ValueError(f'Code {code} is assigned to multiple participants')
            result[code] = f'P{person_index:02d}'
    return result


def extract_code(experiment):
    parts = experiment.strip().rsplit('_', 2)
    if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
        raise ValueError(
            f'Invalid Experiment: {experiment!r}; '
            'expected prefix_<code>_<repeat>'
        )
    return f'{parts[1]}_{parts[2]}'


def mean_sd(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else math.nan
    return mean, sd


def format_mean_sd(values):
    mean, sd = mean_sd(values)
    return f'{mean:.1f}±{sd:.1f}' if math.isfinite(sd) else f'{mean:.1f}±N/A'


def print_table(rows):
    columns = ['Label', 'N'] + METRICS
    widths = {
        col: max(len(col), *(len(str(row[col])) for row in rows))
        for col in columns
    }
    header = '  '.join(f'{col:>{widths[col]}}' for col in columns)
    print(header)
    print('-' * len(header))
    for row in rows:
        print('  '.join(f'{str(row[col]):>{widths[col]}}' for col in columns))


def main(tmp_str,input_path):
    input_path = Path(input_path)
    if input_path.is_dir() or input_path.suffix.lower() != '.csv':
        input_path = input_path / 'topk_per_trial.csv'

    if tmp_str == 'total':
        PARTICIPANT_GROUPS = [
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
    else:
        exit()
        PARTICIPANT_GROUPS = [[1], [2], [3], [4], [5], [6],[7], [8], [9], [10]]
    code_map = build_code_map(PARTICIPANT_GROUPS)
    by_person = defaultdict(lambda: {metric: [] for metric in METRICS})
    by_trial = {metric: [] for metric in METRICS}
    person_codes = defaultdict(set)
    trial_count = 0

    with input_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = {'Label', 'Experiment', 'Trial', *METRICS} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        for line_number, row in enumerate(reader, start=2):
            try:
                code = extract_code(row['Experiment'])
                if code not in code_map:
                    raise ValueError(f'Code {code} not configured in PARTICIPANT_GROUPS')
                if not re.fullmatch(r'[0-9]+', row['Label'].strip()):
                    raise ValueError('Label must contain one or more digits')
                values = {metric: float(row[metric]) for metric in METRICS}
                if any(not math.isfinite(v) or not 0 <= v <= 100 for v in values.values()):
                    raise ValueError('All metric values must be finite percentages from 0 to 100')
            except (ValueError, TypeError, KeyError) as e:
                raise ValueError(f'CSV line {line_number}: {e}') from e
            person = code_map[code]
            for metric, value in values.items():
                by_person[person][metric].append(value)
                by_trial[metric].append(value)
            person_codes[person].add(code)
            trial_count += 1

    if not trial_count:
        raise ValueError('No valid trials in CSV')

    rows = []
    by_participant_means = {metric: [] for metric in METRICS}
    for person in sorted(by_person):
        means = {metric: statistics.mean(by_person[person][metric]) for metric in METRICS}
        for metric in METRICS:
            by_participant_means[metric].append(means[metric])


    if tmp_str == 'quest3':
        EXPECTED_TOTAL_TRIALS = 200
        rows.append({
            'Label': 'OVERALL',
            'N': EXPECTED_TOTAL_TRIALS,
            **{
                metric: format_mean_sd(
                    by_trial[metric] +
                    [0.0] * (EXPECTED_TOTAL_TRIALS - len(by_trial[metric]))
                )
                for metric in METRICS
            },
        })
    else:
        rows.append({
            'Label': 'OVERALL', 'N': trial_count,
            **{metric: format_mean_sd(by_trial[metric]) for metric in METRICS},
        })

    rows.append({
        'Label': 'PERSON-MEAN', 'N': len(by_person),
        **{metric: format_mean_sd(by_participant_means[metric]) for metric in METRICS},
    })

    print(f'Input: {input_path}')
    print_table(rows)


if __name__ == '__main__':
    input = Path('shareleaks_results/results_total')
    tmp_str = 'total'
    main(tmp_str, input)
