# S11 Edge Classification Project

Edge-oriented classification of single-port S11 measurements for classifying surface states (`dry`, `wet`, `ice`).

## Methodology

This repository follows the structured methodology from the internal technical report. The dataset consists of 1,603 S11 measurements (4001 frequency points each, Cartesian representation).

### Data Pipeline
- **Parsing:** Enforces strictly 4001 frequency points per trace and validates grid alignment.
- **Validation:** Outlier detection using 4-sigma deviations across frequency bins.
- **Preprocessing:** Z-score normalization fitted exclusively on the training fold to prevent data leakage. Stratified `70/15/15` split.
- **Augmentation:** Synthetic data generation using physics-aware techniques (Additive Noise, Magnitude Scaling, Frequency Jitter, Phase Perturbation).

### Representation Routes
- **Route A (1D Spectral):** Direct processing of Cartesian vectors to preserve mathematical continuity and prevent phase-wrap discontinuities.
- **Route B (2D Image):** Rasterization of frequency sweeps for 2D CNN classification.

## Experimental Results

The initial baseline evaluation using **Route A (Flattened Cartesian, 1D Spectral)** demonstrates that the physical states are perfectly separable:

### Level 0: Unsupervised Probe
* **Algorithm:** K-Means
* **Train V-Measure:** `1.0000`
* **Validation V-Measure:** `1.0000`
* **Conclusion:** The intrinsic dielectric properties of the three physical states form perfectly distinct mathematical clusters without requiring supervised labels.

### Level 1: Linear Baselines
* **Algorithms:** Logistic Regression & Linear SVM
* **Train Macro F1:** `1.0000`
* **Validation Macro F1:** `1.0000`
* **Conclusion:** The classes are perfectly linearly separable. Given the strict deployment constraints (microcontroller edge hardware), these low-capacity models (~12,000 parameters) are ideal candidates as they easily exceed the 0.85 Macro F1 threshold while requiring a negligible computational footprint.

## Directory Structure
```text
├── configs/            # Experiment YAML configurations
├── data/               # Processed data, scalers, and logs
├── data_exp (1)/       # Raw touchstone files (.s1p)
├── docs/               # Technical methodology reports
├── scripts/cedia/      # SLURM launchers and HPC sync tools
├── src/
│   ├── route_a_1d_spectral/  # Route A models
│   ├── route_b_2d_image/     # Route B models
│   ├── shared_models/        # Common architectures
│   └── data/                 # Parsers, preprocessors, augmentors
```

## HPC Integration
The project natively supports execution on CEDIA's SLURM cluster. See `scripts/cedia` for the deployment manifests and sync tools.

## Contributors
- **Mateo Gavilanes** (mateisaacgavilanes@gmail.com)
- **Yehudhah Kennedy Rodriguez Moran**
