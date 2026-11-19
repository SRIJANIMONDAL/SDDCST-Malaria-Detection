# Malaria Parasite Detection 

## Overview

This repository provides a general-purpose framework for malaria parasite detection in microscopic blood-smear images using YOLO-based object detection models.

The project focuses on dataset preparation, model training, and performance evaluation for detecting and classifying malaria parasite stages.

## Objectives

- Prepare annotated microscopy datasets for object detection.
- Train YOLO-based models on malaria microscopy images.
- Evaluate detection performance using standard object detection metrics.
- Support experimentation with different YOLO architectures.

## Parasite Classes

The framework considers four malaria parasite stages:

1. Ring
2. Trophozoite
3. Gametocyte
4. Schizont

## Dataset Preparation

The project supports datasets organized in YOLO annotation format.

Each image has a corresponding annotation file containing class labels and normalized bounding-box coordinates.

Dataset preparation includes organizing images and annotations into training, validation, and testing subsets.

## Model Training

The framework supports YOLO-based object detection models using the Ultralytics library.

Training configurations can be adjusted according to the available dataset, hardware, and experimental requirements.

## Model Evaluation

Detection performance can be assessed using standard metrics:

- Precision
- Recall
- mAP@50
- mAP@50–95

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Data Availability

Microscopy datasets and annotations are not redistributed in this repository. Users must obtain the necessary data from authorized sources.

## Project Status

This repository is intended for general-purpose malaria parasite detection experiments and educational use.

Additional research implementations and experimental components are maintained separately and are not included in this public repository.

## Acknowledgements

The authors acknowledge the institutional and research support provided by IIT Goa and ICMR–National Institute of Malaria Research (NIMR), Goa.
