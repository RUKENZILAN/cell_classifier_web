# Agentic Cell Culture Classifier

A multi-agent image classification system that classifies cell culture images across different magnification levels (x5, x20, …) and **autonomously decides which magnification level performs best**. Can be run both from the command line and through a Flask-based web interface.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Data Folder Structure](#data-folder-structure)
- [Usage](#usage)
  - [Web interface](#web-interface)
  - [Command line](#command-line)
- [Configuration Parameters](#configuration-parameters)
- [Dummy vs. Real Backbone](#dummy-vs-real-backbone)
- [Outputs](#outputs)
- [Web API Endpoints](#web-api-endpoints)
- [Generating Sample Data](#generating-sample-data)
- [Troubleshooting](#troubleshooting)

## Overview

The system is composed of several agents, each with a specific responsibility, communicating through a shared Central Memory:

1. Reads images, resizes and normalizes them, and splits them into train/test sets.
2. Extracts features using either a pretrained CNN (ResNet-50 / VGG-16) or a fast "dummy" extractor.
3. Classifies with K-Nearest Neighbors (KNN) and computes metrics (accuracy, F1, confusion matrix, per-class results).
4. Optionally narrows confidence intervals using k-fold cross-validation.
5. Compares magnification levels against each other and autonomously decides which one performs best, producing a final report (`final_report.md`).

## Architecture

```
CentralAgent (orchestrator)
 ├── IntakerAgent       → reads data, resizes, splits into train/test
 ├── AnalyzerAgent      → feature extraction + KNN classification + metrics
 └── DecisionMakerAgent → cross-magnification comparison and decision

CentralMemory  → thread-safe shared state read/written by all agents (logs, errors, metrics, results)
```

All core logic lives in a **single file**: `agentic_cell_classifier.py`. `app.py` imports this file and adds a Flask web interface and REST API on top of it.

## Installation

```bash
# Core dependencies
pip install flask numpy scikit-learn pillow

# Real CNN backbones (ResNet-50 / VGG-16) — optional but recommended
pip install torch torchvision
```

> If you don't want to install torch/torchvision, you can still try the system with the **dummy** feature extractor (see [Dummy vs. Real Backbone](#dummy-vs-real-backbone)). This is useful for quickly testing the mechanism, but it does not reflect real classification performance.

## Data Folder Structure

Your data folder should be organized as follows — one subfolder per magnification level, and under each of those, one subfolder per class:

```
data/
  x5/
    Healthy_Cell/*.jpg
    Diseased_Cell/*.jpg
  x20/
    Healthy_Cell/*.jpg
    Diseased_Cell/*.jpg
```

Supported file extensions: `.png .jpg .jpeg .tif .tiff .bmp`

If you don't have your own data yet, you can generate a synthetic test dataset using the `generate_sample_data.py` script (see [Generating Sample Data](#generating-sample-data)).

## Usage

### Web interface

```bash
python app.py
# Open in browser: http://127.0.0.1:5000
```

From the interface:

1. **1. Select Data** — pick your `data/` folder (via the local file picker) or type a server-side path directly.
2. **2. Configure** — set magnification levels, backbone, KNN k value, test size, output folder, and cross-validation fold count.
3. **▶ Run Pipeline** — run it; live logs, metrics, confusion matrix, per-class results, misclassified examples, and the final report update in real time in the interface.
4. Export results as PNG/CSV/JSON/Markdown, or download all visuals as a single ZIP.

### Command line

**Interactive mode:**
```bash
python agentic_cell_classifier.py
```
You'll be asked for the data folder and whether torch/torchvision is installed.

**With direct parameters:**
```bash
python agentic_cell_classifier.py --data-dir ./data --magnifications x5 x20 --backbone resnet50
```

To run in dummy (no-torch) mode:
```bash
python agentic_cell_classifier.py --data-dir ./data --dummy
```

## Configuration Parameters

| Parameter | Description | Default |
|---|---|---|
| `data_dir` | Data folder path | — (required) |
| `magnifications` | Magnification levels to compare | `["x5", "x20"]` |
| `backbone` | `resnet50` \| `vgg16` \| `dummy` | `resnet50` |
| `use_dummy_extractor` | Fast test mode without torch | `False` |
| `knn_neighbors` | Number of KNN neighbors (k) | `5` |
| `knn_metric` | KNN distance metric | `cosine` |
| `test_size` | Test set fraction (between 0–1) | `0.2` |
| `cv_folds` | Number of k-fold cross-validation folds (0 = disabled) | `0` |
| `output_dir` | Folder where reports and outputs are written | `./outputs` |
| `random_state` | Fixed seed for reproducibility | `42` |
| `max_misclassified_examples` | Max misclassified examples shown per gallery | `8` |

> **Statistical note:** With small test sets (roughly 20–30 examples per class), confidence intervals can overlap between magnifications. For a more reliable comparison, set `cv_folds` to 5 or 10 to use k-fold cross-validation — this narrows confidence intervals using the same dataset, without collecting additional data.

## Dummy vs. Real Backbone

| | **Dummy** | **ResNet-50 / VGG-16** |
|---|---|---|
| Dependency | numpy only | torch + torchvision |
| Features | Pixel channel mean/std + histogram | Deep CNN features pretrained on ImageNet |
| Speed | Very fast | Slow (CPU) / fast (GPU) |
| Purpose | Testing the mechanism, development | Real classification performance |
| Accuracy | Does not reflect real performance | Reflects real performance |

To switch to a real backbone:
1. `pip install torch torchvision`
2. In the web interface, uncheck **"Force dummy extractor"** and select `ResNet-50` or `VGG-16` as the **Backbone**.
3. In the CLI, use `--backbone resnet50` (or `vgg16`) and **do not** add the `--dummy` flag.

On first run, pretrained ImageNet weights are downloaded from the internet (a few hundred MB); subsequent runs read from the local cache (`~/.cache/torch`).

## Outputs

Written to the specified `output_dir` (default `./outputs`):

- `final_report.md` — a readable summary report (metrics, decision, rationale)
- `run_memory.json` — the full state record of the run (logs, errors, raw metrics)
- Confusion matrices, accuracy/F1 charts, and misclassified example galleries (downloadable as PNG/ZIP from the web interface)

## Web API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web interface |
| `/api/upload` | POST | Upload a local folder |
| `/api/run` | POST | Starts the pipeline |
| `/api/logs` | GET | Live log stream (SSE) |
| `/api/status` | GET | Current run status |
| `/api/cancel` | POST | Cancels the running job |
| `/api/resolve-output` | POST | Resolves the output folder path |
| `/api/export/zip` | GET | Downloads all visuals as a ZIP |
| `/api/export/metrics.csv` | GET | Downloads metrics as CSV |
| `/api/export/results.json` | GET | Downloads results as JSON |
| `/api/export/report.md` | GET | Downloads the final report as Markdown |
| `/api/reset` | POST | Resets the interface and state |

## Generating Sample Data

If you don't have your own data, you can generate a small synthetic dataset to try out the interface/pipeline:

```bash
python generate_sample_data.py
```

This creates 12 synthetic images per class (colored circles) under `./data/x5/` and `./data/x20/`, in `class_a` / `class_b` subfolders.

## Troubleshooting

| Issue | Likely cause / fix |
|---|---|
| "Choose a folder" or the dark/light theme button isn't clickable | There may be a JS initialization error in `index.html`; reload the page and check the browser console for errors. |
| `ModuleNotFoundError: torch` | Run `pip install torch torchvision`, or use the `use_dummy` / `--dummy` option. |
| Confidence intervals overlap | Use cross-validation with `cv_folds` ≥ 5, or increase the test set size (see the note under Configuration Parameters). |
| Timeout when uploading large folders | Use a server-side path (`data_dir`) to skip the upload step. |

---

*This project is an educational/research reference implementation; it should not be used for clinical diagnosis.*
