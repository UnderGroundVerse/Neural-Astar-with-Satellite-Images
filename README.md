# Neural A* with Satellite Images

A hybrid pathfinding project that combines:
- **road extraction from satellite imagery** (CNN segmentation), and
- **A\*** search on obstacle grids, including a **learned neural heuristic**.

The goal is to generate traversable maps from real imagery and then perform efficient route planning.

---

## Project Overview

This repository contains two linked subsystems:

1. **Road Segmentation**
   - `RoadExtractor.py` defines a custom Keras model that predicts road masks from satellite images.
   - `dataset.py` provides dataset loading and preprocessing utilities for images and masks.
   - `training.py` includes model training and GPU setup helpers.
   - `notebooks/RoadExtractor.ipynb` demonstrates experimentation/training.

2. **Pathfinding + Learned Heuristic**
   - `astar.py` implements classic A* with Manhattan/Euclidean heuristic.
   - `DifferentialAstar` (in `astar.py`) swaps the hand-crafted heuristic with a neural predictor.
   - `heuristic_model.py` defines a feed-forward model to regress path cost heuristic.
   - `data_generation.py` creates synthetic blocked-grid datasets for heuristic training.
   - `feed_forward_approch.ipynb` explores this training flow.

---

## Repository Structure

- `RoadExtractor.py` – CNN-style road segmentation model (`keras.Model`)
- `dataset.py` – image/mask loading, resizing, normalization, binarization
- `training.py` – training helpers, optional GPU memory growth setup
- `astar.py` – A* implementation + visualization + neural-heuristic extension
- `heuristic_model.py` – feed-forward heuristic model creation/load/save
- `data_generation.py` – synthetic grid dataset generation and pickle persistence
- `feed_forward_approch.ipynb` – notebook for heuristic model experiments
- `notebooks/` – additional research notebooks
- `trained_models/` – pre-trained model weights/artifacts

---

## Core Concepts

### 1) A* Search
The `Astar` class performs 4-neighbor grid search on a binary obstacle map:
- `0`: free cell
- `1`: blocked cell

It computes:
- `g`: path cost from start
- `h`: heuristic estimate to goal
- `f = g + h`

### 2) Differential / Neural A*
`DifferentialAstar` extends `Astar` and overrides `calc_h`:
- flatten current grid
- concatenate `[grid_flat, current_position, goal_position]`
- predict heuristic using a neural model

This replaces fixed distance formulas with a learned estimate.

### 3) Road Segmentation
`RoadExtractor` predicts a single-channel road mask from RGB satellite input.
Typical downstream usage:
- threshold predicted mask
- convert mask into occupancy grid
- run A* over the resulting traversability map

---

## Setup

> This project currently does not include a pinned dependency file (`requirements.txt` or `pyproject.toml`), so install manually.

Recommended environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install tensorflow keras numpy pandas pillow opencv-python matplotlib scikit-learn
```

---

## Quick Start

### A) Classic A* on a toy grid

```bash
python astar.py
```

This runs the example in `__main__`, finds a path, and visualizes it.

### B) Train a heuristic model (synthetic data path)

Use `data_generation.py` and `heuristic_model.py` from a script or notebook:

```python
from data_generation import GenerateData
from heuristic_model import HeuristicModel

X, y = GenerateData.generate_training_data(num_samples=1000, grid_size=(20, 20))
model = HeuristicModel.create_feed_forward_heuristic_model(input_dim=X.shape[1])
model.fit(X, y, epochs=10, batch_size=32)
HeuristicModel.save_model(model, "feed_forward_n1000_s20x20_e10_bs32")
```

### C) Road extractor training/inference

Use:
- `dataset.py` to load/prepare image+mask tensors
- `RoadExtractor.py` to build/compile model
- `training.py` to run `.fit(...)`
- `notebooks/RoadExtractor.ipynb` for an end-to-end exploratory workflow

---

## Data Expectations

`dataset.py` expects metadata CSVs with at least:
- `sat_image_path`
- `mask_path`

Images are resized to a default `(256, 256)` and normalized to `[0,1]`.
Masks are grayscale-converted and thresholded for binary occupancy use.

---

## Pretrained Models

Available in `trained_models/`:
- `RoadExtractor-weights-118ep.h5`
- `feed_forward_n1000_s20x20_e10_bs32.keras`

---

## Notes / Current Limitations

- No dependency lock file is provided.
- Some helper methods are class methods without `self` and are intended to be called as utility functions.
- `data_generation.py` contains duplicate definitions of `generate_training_data`; the latter overrides the former.
- The repo is notebook-heavy (research/prototyping style) and can benefit from packaging and script entrypoints.

---

## Suggested Next Improvements

1. Add `requirements.txt` (or `pyproject.toml`) with exact versions.
2. Add a reproducible training script for road segmentation and neural heuristic.
3. Add evaluation metrics:
   - segmentation IoU/F1
   - A* node expansions/time with classic vs neural heuristic
4. Add unit tests for:
   - grid validity/block checks
   - deterministic path outputs on fixed maps
   - heuristic model input/output shape checks
5. Standardize model save/load APIs and naming conventions.

---

## License

No license file is currently present in the repository.  
Add one (e.g., MIT) if you intend others to reuse or modify this code.
