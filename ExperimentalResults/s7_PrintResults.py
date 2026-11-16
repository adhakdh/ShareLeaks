
# -*- coding: utf-8 -*-
"""Compact ShareLeaks Top-K summary.

Reports OVERALL and PERSON-MEAN statistics.
All metric values are percentages (0–100).
Standard deviations use the sample SD.

Participant identification:
    user1_1  -> P01
    user1_5  -> P01
    user2_1  -> P02
    user22_5 -> P22
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


def extract_participant(experiment, tmp_str):
    """Extract participant ID from experiment name."""

    experiment = experiment.strip()

    if tmp_str == 'total':
        match = re.fullmatch(
            r'user(\d+)_(\d+)',
            experiment
        )

        if not match:
            raise ValueError(
                f'Invalid Experiment: {experiment!r}; '
                'expected user<participant>_<repeat>'
            )

        person_index = int(match.group(1))
        repeat_index = int(match.group(2))

        if not (1 <= person_index <= 22):
            raise ValueError(
                f'Invalid participant index: {person_index}'
            )

        if not (1 <= repeat_index <= 5):
            raise ValueError(
                f'Invalid repeat index: {repeat_index}'
            )

        return f'P{person_index:02d}'

    if tmp_str == 'quest3':
        # Preserve the original Quest 3 naming convention.
        match = re.fullmatch(
            r'.*_(\d+)_(\d+)',
            experiment
        )

        if not match:
            raise ValueError(
                f'Invalid Quest3 Experiment: {experiment!r}'
            )

        return f'P{int(match.group(1)):02d}'

    raise ValueError(f'Unsupported dataset: {tmp_str}')


def mean_sd(values):
    """Calculate mean and sample standard deviation."""

    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else math.nan

    return mean, sd


def format_mean_sd(values):
    """Format mean ± sample SD."""

    mean, sd = mean_sd(values)

    if math.isfinite(sd):
        return f'{mean:.1f}±{sd:.1f}'

    return f'{mean:.1f}±N/A'


def print_table(rows, metrics):
    """Print a compact aligned results table."""

    columns = ['Label', 'N'] + metrics

    widths = {
        col: max(
            len(col),
            *(len(str(row[col])) for row in rows)
        )
        for col in columns
    }

    header = '  '.join(
        f'{col:>{widths[col]}}'
        for col in columns
    )

    print(header)
    print('-' * len(header))

    for row in rows:
        print(
            '  '.join(
                f'{str(row[col]):>{widths[col]}}'
                for col in columns
            )
        )


def read_results(input_path, tmp_str, metrics):
    """Read trial-level results and group them by participant."""

    by_person = defaultdict(
        lambda: {metric: [] for metric in metrics}
    )

    by_trial = {
        metric: [] for metric in metrics
    }

    trial_count = 0

    with input_path.open(
        'r',
        encoding='utf-8-sig',
        newline=''
    ) as f:

        reader = csv.DictReader(f)

        required = {
            'Label',
            'Experiment',
            'Trial',
            *metrics,
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f'Missing columns: {sorted(missing)}'
            )

        for line_number, row in enumerate(reader, start=2):

            try:
                person = extract_participant(
                    row['Experiment'],
                    tmp_str
                )

                if not re.fullmatch(
                    r'[0-9]+',
                    row['Label'].strip()
                ):
                    raise ValueError(
                        'Label must contain one or more digits'
                    )

                values = {
                    metric: float(row[metric])
                    for metric in metrics
                }

                if any(
                    not math.isfinite(value)
                    or not 0 <= value <= 100
                    for value in values.values()
                ):
                    raise ValueError(
                        'All metric values must be finite '
                        'percentages from 0 to 100'
                    )

            except (ValueError, TypeError, KeyError) as e:
                raise ValueError(
                    f'CSV line {line_number}: {e}'
                ) from e

            for metric, value in values.items():
                by_person[person][metric].append(value)
                by_trial[metric].append(value)

            trial_count += 1

    if not trial_count:
        raise ValueError('No valid trials in CSV')

    return by_person, by_trial, trial_count


def calculate_summary(
    tmp_str,
    by_person,
    by_trial,
    trial_count,
    metrics,
):
    """Calculate OVERALL and PERSON-MEAN statistics."""

    rows = []

    by_participant_means = {
        metric: [] for metric in metrics
    }

    for person in sorted(by_person):

        for metric in metrics:
            values = by_person[person][metric]

            by_participant_means[metric].append(
                statistics.mean(values)
            )

    # Trial-level overall statistics
    if tmp_str == 'quest3':
        expected_total_trials = 200

        if trial_count > expected_total_trials:
            raise ValueError(
                f'Trial count {trial_count} exceeds '
                f'expected total {expected_total_trials}'
            )

        overall_values = {
            metric: by_trial[metric]
            + [0.0] * (expected_total_trials - trial_count)
            for metric in metrics
        }

        overall_n = expected_total_trials

    else:
        overall_values = by_trial
        overall_n = trial_count

    rows.append({
        'Label': 'OVERALL',
        'N': overall_n,
        **{
            metric: format_mean_sd(overall_values[metric])
            for metric in metrics
        },
    })

    # Participant-level macro statistics
    rows.append({
        'Label': 'PERSON-MEAN',
        'N': len(by_person),
        **{
            metric: format_mean_sd(
                by_participant_means[metric]
            )
            for metric in metrics
        },
    })

    return rows


def main(tmp_str, input_path):
    """Run the complete Top-K summary pipeline."""

    metrics = [
        'C@1', 'C@3',
        'S@1', 'S@3', 'S@5',
        'S@10', 'S@20', 'S@30',
        'S@50', 'S@100',
    ]

    input_path = Path(input_path)

    if input_path.suffix.lower() != '.csv':
        input_path = input_path / 'topk_per_trial.csv'

    # Read trial-level results
    by_person, by_trial, trial_count = read_results(
        input_path,
        tmp_str,
        metrics,
    )

    # Calculate summary statistics
    rows = calculate_summary(
        tmp_str,
        by_person,
        by_trial,
        trial_count,
        metrics,
    )

    # Print results
    print(f'Input: {input_path}')
    print_table(rows, metrics)


if __name__ == '__main__':

    input_path = Path('shareleaks_results/results_total')
    tmp_str = 'total'

    main(tmp_str, input_path)
