# Route B: 2D Image-Like Representations

## Overview
This directory contains the data rendering and modelling code dedicated to **Route B**. This route converts the 1D spectral traces into two-dimensional image representations, enabling the use of established 2D Convolutional Neural Networks (CNNs) designed for computer vision tasks.

## Data Encoding
The conversion introduces a rendering step that rasterises the continuous-valued amplitude vs. frequency curves into discrete pixel grids (e.g., `224 x 224` pixels).

Primary encodings explored:
1. **B1: Grayscale Curve Images**: A direct plot of the spectral trace against frequency on a blank canvas.
2. **B2: Dual-Channel Rasterised Maps**: Real and Imaginary components rendered as separate image channels.
3. **B3: Spectrogram/Heatmap**: Frequency-axis interpolation or recurrence plot generation to encode spectral correlations as texture.

## Submodules
- `data_rendering/`: Scripts that handle the conversion of the `dataset_normalized.npz` 1D vectors into standardised image tensors. Strict standardisation of line thickness, axis scaling, and anti-aliasing is enforced here.
- `models/`: Contains the 2D CNN architectures (e.g., Custom 2D CNNs, pre-trained transfer learning models like MobileNet or ResNet-50).

## Artefact Warning
When evaluating Route B models, it is critical to compare them against Route A baselines. Image rendering can introduce artefacts (e.g., sub-pixel smoothing gradients, bounding box scales) that a neural network might learn to exploit as confounding variables, which do not exist in the raw physical measurement.
