# MS-MRI-Classifier: Multi-Planar Deep Transfer Learning for Multiple Sclerosis Lesion Detection

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning diagnostic pipeline for automated detection of Multiple Sclerosis (MS) demyelinating lesions from brain Magnetic Resonance Imaging (MRI) volumes. Built with TensorFlow, this framework emphasizes clinical integrity by enforcing patient-level data partitioning (zero inter-slice leakage), multi-planar spatial sampling (Axial, Coronal, Sagittal), dynamic learning rate decay, and rigorous contrast-agent (Gadolinium) ablation benchmarking.

---

## Key Engineering & Clinical Highlights

* **Patient-Level Data Partitioning (Leakage-Free):** Slices are partitioned strictly at the patient/participant level before extraction, preventing identical brain morphology from leaking across training and evaluation splits.
* **Multi-Planar Volumetric Sampling:** Volumetric extraction across **Axial**, **Coronal**, and **Sagittal** anatomical planes, coupled with adaptive 15–20% boundary-slice exclusion to remove non-brain cranial margins.
* **Contrast Agent Ablation Study:** Quantitative clinical comparison measuring model diagnostic robustness with and without Gadolinium (GADO) enhancement across multi-center cohorts (MICCAI 2016, ISBI, and Baghdad 2022).
* **High-Throughput Streaming Pipeline:** Custom `tf.data` pipeline with on-the-fly spatial augmentations (90° rotations, horizontal/vertical reflections), channel expansion, and background-safe intensity scaling.
* **Comprehensive Diagnostic Metrics:** Evaluated through Precision-Recall Curves, ROC-AUC (micro/macro), and Confusion Matrices alongside standard Accuracy to ensure safety in class-skewed regimes.

---

## Experimental Benchmark Results

### 1. Multi-Planar Analysis (Sagittal + Coronal + Axial)
Evaluated across 78,000+ balanced multi-view slices with patient-level separation:

| Model Backbone | Test Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VGG16 Transfer** | **80.0%** | **0.80** | **0.81** | **0.80** | **0.87** |
| **InceptionV3** | 78.0% | 0.78 | 0.83 | 0.81 | 0.88 |

### 2. Contrast Agent Ablation (With GADO vs. Without GADO)
Comparative assessment on Sagittal sequences showing model viability even when contrast agent administration is clinically contraindicated:

| Architecture | Cohort Type | Test Accuracy | Test Loss | Recall | Precision | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VGG16** | With GADO | 78.0% | 0.43 | 0.92 | 0.71 | 0.80 |
| **VGG16** | Without GADO | 78.0% | 0.44 | 0.91 | 0.72 | 0.80 |
| **InceptionV3** | With GADO | 78.0% | 0.41 | 0.86 | 0.73 | 0.79 |
| **InceptionV3** | Without GADO | 77.0% | 0.43 | 0.77 | 0.73 | 0.75 |

---

## Repository Structure

```text
ms-mri-classifier/
├── config/
│   └── default_config.yaml     # Training parameters and dataset splitting ratios
├── src/
│   ├── dataset_prep.py         # NIfTI loading, volume slicing, safe normalization
│   ├── lookup_builder.py       # Patient-level splitting and balanced CSV export
│   ├── tf_pipeline.py          # High-performance tf.data pipelines with augmentations
│   ├── models.py               # Transfer learning heads for VGG16 and InceptionV3
│   └── evaluate.py             # ROC, PR curves, and clinical confusion matrices
├── examples/
│   └── train_and_evaluate.py   # Complete training and evaluation entry point
├── tests/
│   └── test_pipeline.py        # Automated Pytest suite for pipeline integrity
├── requirements.txt
├── .gitignore
└── README.md
```
