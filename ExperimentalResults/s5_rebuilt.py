
# -*- coding: utf-8 -*-
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
