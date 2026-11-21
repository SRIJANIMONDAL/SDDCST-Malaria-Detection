# Malaria Parasite Detection and Stage Classification Using YOLO

A Python-based framework for automated malaria parasite detection and stage classification in microscopic blood-smear images using YOLO-based object detection models.

This repository provides utilities for dataset preparation, model training, and performance evaluation, supporting experiments with different YOLO architectures.

## Overview

Malaria parasite detection using microscopic blood-smear images is an important application of computer vision in medical image analysis.

Deep learning-based object detection models, particularly the YOLO family, provide an efficient approach to identifying and classifying malaria parasites in microscopy images.

This repository provides a modular workflow covering:

- Preparation of annotated microscopy datasets
- Validation of YOLO-format annotations
- Training, validation, and testing dataset organization
- Training with configurable YOLO architectures
- Checkpoint-based training resumption
- Detection performance evaluation
- Overall and class-wise performance reporting

The provided utilities can be adapted to microscopy datasets organized in standard YOLO annotation format.

## Parasite Classes

The framework considers four malaria parasite developmental stages:

| Class ID | Parasite Stage |
|---|---|
| 0 | Ring |
| 1 | Trophozoite |
| 2 | Gametocyte |
| 3 | Schizont |

The dataset preparation script also supports other class configurations through command-line arguments.

## Repository Structure

```text
SDDCST-Malaria-Detection/
├── .gitignore
├── README.md
├── evaluate_v1_prime.py
├── requirements.txt
├── split.py
└── train_dda_v1_prime_combinations.py
```

## Script Descriptions

| File | Description |
|---|---|
| `split.py` | Dataset preparation, annotation validation, and train/validation/test splitting |
| `train_dda_v1_prime_combinations.py` | YOLO model training with configurable architectures and training parameters |
| `evaluate_v1_prime.py` | Model evaluation with overall and class-wise detection metrics |
| `requirements.txt` | Python dependencies required for the framework |
| `.gitignore` | Excludes unnecessary local files and generated outputs from version control |

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SRIJANIMONDAL/SDDCST-Malaria-Detection.git
cd SDDCST-Malaria-Detection
```

### 2. Install Dependencies

Python 3.10 or later is recommended, subject to library compatibility.

```bash
pip install -r requirements.txt
```

The framework uses:

- Python
- PyTorch
- Ultralytics YOLO
- PyYAML

GPU acceleration can be used with a compatible CUDA-enabled PyTorch installation.

## Dataset Preparation

The framework supports microscopy datasets with annotations in YOLO object detection format.

### Input Dataset Structure

```text
dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── labels/
    ├── image_001.txt
    ├── image_002.txt
    └── ...
```

Each image must have a corresponding annotation file containing class labels and normalized bounding-box coordinates.

YOLO annotations follow the format:

```text
class_id x_center y_center width height
```

### Dataset Splitting

The `split.py` utility performs reproducible image-level random splitting and generates the required YOLO dataset structure.

```bash
python split.py --root dataset --output dataset_split --classes Ring Trophozoite Gametocyte Schizont
```

Default dataset proportions:

| Dataset Subset | Percentage |
|---|---|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Custom split ratios can also be specified:

```bash
python split.py --root dataset --output dataset_split --classes Ring Trophozoite Gametocyte Schizont --train-ratio 0.80 --val-ratio 0.10 --test-ratio 0.10
```

### Generated Dataset Structure

```text
dataset_split/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

The script automatically generates `data.yaml` containing dataset paths and class information.

**Note:** The utility performs image-level random splitting. For datasets containing multiple images from the same patient or specimen, group-aware splitting may be necessary to avoid data leakage.

## Model Training

The `train_dda_v1_prime_combinations.py` script supports training YOLO object detection models using the Ultralytics framework.

### Training Features

- Configurable YOLO model architectures
- Multiple model training
- Custom training epochs
- Configurable image size and batch size
- CPU and GPU execution
- Checkpoint-based training resumption
- Configurable optimizer and learning rate
- Reproducible training configurations

### Train a YOLO Model

```bash
python train_dda_v1_prime_combinations.py --data dataset_split/data.yaml --models yolo11n.pt --epochs 100
```

### Train Multiple Models

```bash
python train_dda_v1_prime_combinations.py --data dataset_split/data.yaml --models yolo11n.pt yolo12n.pt --epochs 100
```

### Custom Training Configuration

```bash
python train_dda_v1_prime_combinations.py --data dataset_split/data.yaml --models yolo11n.pt --epochs 100 --imgsz 640 --batch 16 --device 0
```

### Training Outputs

Trained models and training logs are saved in the configured project directory.

```text
runs/train/
└── yolo11n/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.csv
    └── args.yaml
```

The training utility supports resuming from saved checkpoints. Runs containing existing best-model weights are skipped unless a new run is requested.

## Model Evaluation

The `evaluate_v1_prime.py` script evaluates trained YOLO object detection models on annotated datasets.

### Evaluation Metrics

| Metric | Description |
|---|---|
| Precision | Proportion of predicted detections that are correct |
| Recall | Proportion of ground-truth objects successfully detected |
| mAP@50 | Mean Average Precision at an IoU threshold of 0.50 |
| mAP@50–95 | Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95 |

### Evaluate a Trained Model

```bash
python evaluate_v1_prime.py --weights runs/train/yolo11n/weights/best.pt --data dataset_split/data.yaml --split test
```

### Custom Evaluation Configuration

```bash
python evaluate_v1_prime.py --weights runs/train/yolo11n/weights/best.pt --data dataset_split/data.yaml --split test --imgsz 640 --batch 16 --device 0
```

### Evaluation Outputs

```text
evaluation_results/
├── overall_metrics.csv
├── per_class_metrics.csv
└── validation/
```

The script generates two CSV files:

**Overall Metrics**

- Precision
- Recall
- mAP50
- mAP50–95

**Per-Class Metrics**

- Class ID
- Class name
- Precision
- Recall
- AP50
- AP50–95

These metrics can be used to analyze detection performance across parasite classes and compare different trained models.

## General Workflow

The framework follows a standard YOLO-based object detection workflow:

```text
Microscopic Blood-Smear Images
              |
              v
      YOLO Annotations
              |
              v
      Dataset Preparation
           split.py
              |
              v
     Train / Val / Test
              |
              v
        Model Training
train_dda_v1_prime_combinations.py
              |
              v
       Trained YOLO Model
              |
              v
        Model Evaluation
      evaluate_v1_prime.py
              |
              v
   Overall and Class-wise Metrics
```

## Data Availability

Microscopy datasets and corresponding annotations are not redistributed through this repository.

Users must obtain the necessary datasets from their respective authorized sources and comply with applicable data usage policies.

## Acknowledgements

The authors acknowledge the institutional and research support provided by the Indian Institute of Technology Goa (IIT Goa) and the ICMR–National Institute of Malaria Research (NIMR), Goa.
