\# SDDCST: Stochastic Dual-Domain Color Statistics Transfer for Malaria Parasite Detection



\## Overview



This repository contains the implementation of \*\*Stochastic Dual-Domain Color Statistics Transfer (SDDCST)\*\*, a lightweight, annotation-preserving data augmentation strategy investigated for cross-dataset malaria parasite detection.



SDDCST combines Lab color statistics from two reference domains through stochastic interpolation and controlled blending to generate diverse microscopy image appearances while preserving the original parasite bounding-box annotations.



\## Datasets



Three malaria microscopy datasets are used:



| Dataset | Description | Images |

|---|---|---:|

| New\_NIMR (A) | NIMR dataset with multi-instance annotations | 150 |

| Old\_NIMR (B) | NIMR dataset with predominantly single-object annotations | 155 |

| MP\_IDB (C) | Public malaria microscopy dataset | 40 |



Four parasite classes are considered: \*\*Ring, Trophozoite, Gametocyte, and Schizont\*\*.



The NIMR datasets are not redistributed in this repository.



\## Proposed SDDCST Augmentation



1\. Select a source training image and two reference images from the other domains.

2\. Convert the images into CIELAB color space.

3\. Compute channel-wise means and standard deviations.

4\. Interpolate reference statistics using a randomly sampled mixing coefficient.

5\. Transfer the interpolated statistics to the source image.

6\. Blend the transformed image with the original source image.

7\. Preserve the original YOLO bounding-box annotations.



The stochastic parameters are:



\- Lambda: uniformly sampled from 0.2 to 0.8.

\- Alpha: uniformly sampled from 0.4 to 0.8.



Two augmented images are generated for each original training image.



\## Experimental Setup



The augmented-only training experiment uses:



\- New\_NIMR augmented images: 200

\- Old\_NIMR augmented images: 214

\- MP\_IDB augmented images: 52



\*\*Total: 466 augmented training images.\*\*



Original training images are excluded from this experiment. Validation and test images remain unaugmented.



The evaluated detectors are YOLO11n, YOLO12n, and YOLO26n.



\## Repository Files



| File | Description |

|---|---|

| `split.py` | Dataset splitting |

| `dda\_v1\_color\_transfer.py` | SDDCST augmentation implementation |

| `prepare\_dda\_v1\_prime\_combinations.py` | Dataset combination preparation |

| `prepare\_train\_v1\_prime.py` | Augmented-only training preparation |

| `train\_v1\_prime.py` | YOLO training |

| `evaluate\_v1\_prime.py` | Evaluation |

| `train\_dda\_v1\_prime\_combinations.py` | Additional combination experiments |



\## Installation



```bash

pip install -r requirements.txt

```



\## Data Availability



Dataset files are not included. Researchers must obtain the datasets from their authorized sources and organize them according to the paths expected by the scripts.



\## Reproducibility



The scripts reflect the research implementation. Some paths are currently specific to the original experimental environment and must be configured for another machine.



