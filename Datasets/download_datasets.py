
# -*- coding: utf-8 -*-
"""Download the ShareLeaks dataset from Google Drive.

Usage:
    pip install -U gdown
    python download_datasets.py

The script downloads the dataset archive, extracts it
into Datasets/, and preserves the original directory structure.
"""

from pathlib import Path
import zipfile
import gdown


def download_datasets():
    # Google Drive shared file
    url = (
        "https://drive.google.com/file/d/"
        "1RuArdKcFwE7eOoEd4JI-TejL1I6A2qlK/view?usp=drive_link"
    )

    # Output directories
    dataset_root = Path("Datasets")
    archive_path = dataset_root / "shareleaks_dataset.zip"

    dataset_root.mkdir(parents=True, exist_ok=True)

    print("========== ShareLeaks Dataset ==========")
    print(f"Google Drive: {url}")
    print(f"Output: {archive_path}")
    print("========================================")

    # Download archive with resume support
    if not archive_path.exists() or not zipfile.is_zipfile(archive_path):
        print("\nDownloading dataset...")

        result = gdown.download(
            url=url,
            output=str(archive_path),
            quiet=False,
            fuzzy=True,
            resume=True,
        )

        if result is None:
            raise RuntimeError(
                "Dataset download failed. "
                "Please verify the Google Drive sharing permissions."
            )

    else:
        print("\nDataset archive already exists. Skipping download.")

    # Verify ZIP archive
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(
            f"Downloaded file is not a valid ZIP archive: {archive_path}"
        )

    # Extract archive
    print("\nExtracting dataset...")

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(dataset_root)

    print("\n========== Completed ==========")
    print(f"Dataset directory: {dataset_root}")
    print(f"Archive: {archive_path}")
    print("===============================")


if __name__ == "__main__":
    download_datasets()
