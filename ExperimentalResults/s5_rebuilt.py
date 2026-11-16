
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
from shareleaks_lib import process 



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
