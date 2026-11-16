# ShareLeaks Datasets

The datasets in this repository are provided for the paper **ShareLeaks**.

Due to repository storage limitations, the original XR first-person-view (FPV) recordings are hosted separately on Google Drive. The dataset contains approximately **8 GB of video recordings** used in the main experiments reported in the paper.

## 1. Dataset Download

The complete video dataset is available at Google Drive. The dataset can be downloaded automatically using the provided Python script.

First, install the required dependency:

```bash
pip install -U gdown
```

Then, from the repository root directory, execute:

```bash
python download_datasets.py
```

The script downloads the dataset archive and extracts its contents into the `Datasets/` directory.

## 2. Dataset Structure

After extraction, the video dataset is organized by input category and recording identifier.

```text
Datasets/
└── video/
    ├── 0133/
    │   ├── user1_1.mp4
    │   ├── user1_2.mp4
    │   └── ...
    ├── 0134/
    │   ├── user1_1.mp4
    │   ├── user1_2.mp4
    │   └── ...
    └── ...
```

Each numeric directory, such as `0133`, represents an input category.

Video filenames follow the format:

```text
user<participant_id>_<recording_id>.mp4
```

For example, `user1_2.mp4` represents the second recording associated with Participant 1.

## 3. Reproducing the Experimental Results

The downloaded videos provide the original input data for the ShareLeaks processing pipeline.

The corresponding experimental scripts are located in:

```text
ExperimentalResults/
```

The pipeline consists of seven processing stages, from video frame extraction to evaluation result generation.

To reproduce the main experimental results, execute:

```bash
cd ExperimentalResults
python run.py
```

The repository also includes key intermediate results in:

```text
ExperimentalResults/shareleaks_results/
```

These intermediate results support reproduction of the main evaluation without requiring reviewers to process the complete original video dataset.

## 4. Dataset Scope

The provided dataset contains the original video recordings used in the main ShareLeaks experiments.

The recordings are organized using anonymized participant identifiers and input categories to facilitate reproducibility.

Due to the size of the dataset, the videos are distributed separately from the source code and experimental scripts.

## 5. Additional Information

If the paper is accepted, we will provide the complete experimental results and additional supporting materials associated with the study.

**If you encounter any difficulties downloading the dataset or reproducing the experimental results, please do not hesitate to contact us. Thank you sincerely for your interest in ShareLeaks and for your time in evaluating our artifact.**