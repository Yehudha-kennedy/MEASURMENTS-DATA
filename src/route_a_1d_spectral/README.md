# Route A: 1D Spectral Representations

## Overview
This directory contains all modelling code dedicated to **Route A**, which treats the S11 frequency sweep as a one-dimensional feature vector. This route preserves the full numerical fidelity of the original microwave measurements by avoiding any spatial rasterisation.

## Data Encoding
We exclusively use **Cartesian representation** (Real and Imaginary components) to avoid the artificial phase-wrapping discontinuities inherent to polar (Magnitude/Phase) formats.

Three primary encodings are supported by the models in this directory:
1. **A1: Dual-Channel Cartesian Vector** (`2 x 4001`): The canonical format. Suitable for 1D CNNs and ResNet-1D architectures.
2. **A2: Flattened Cartesian Vector** (`8002`): The dual channels flattened into a single vector. Used for classical models (Logistic Regression, SVM, Random Forest, XGBoost).
3. **A3: Derivative Augmented**: Includes the first-order finite differences to highlight local spectral features (slopes, inflection points).

## Submodules
- `models/`: Contains the implementations for:
  - **Level 0**: Unsupervised Probe (K-Means)
  - **Level 1**: Linear Baselines (Logistic Regression, Linear SVM)
  - **Level 2**: Tabular Powerhouses (Gradient Boosted Trees, Random Forest)
  - **Level 3**: Deep Architectures (1D CNN, ResNet-1D)

## Advantages of Route A
- **Lossless**: Preserves floating-point precision.
- **Artefact-free**: Avoids line-thickness, anti-aliasing, and background scaling artefacts introduced by image-rendering.
- **Computationally Efficient**: Requires no rendering step during preprocessing.
