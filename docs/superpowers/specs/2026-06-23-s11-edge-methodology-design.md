# S11 Edge Methodology Report Design

**Date:** 2026-06-23

## Goal

Produce a professional PDF report in English that defines a manuscript-grade methodological strategy for classifying `single-port S11` Touchstone `.s1p` measurements from three operating states using edge-oriented machine learning. The report is not allowed to present fabricated experiments, provisional scores, or unverified performance claims. Its role is to help decide the analysis pipeline before manuscript drafting.

## Fixed Scope

- Data type: `single-port S11 only`
- Input files: Touchstone `.s1p`
- Study mode: methodological plan only
- Writing style: internal technical strategy report
- Comparison mode: neutral and comparative
- Representation strategy: dual route
- Route A: `1D spectral`
- Route B: `2D image-like`
- Target deployment envelope: generic edge, broader than strict ESP32-only constraints
- Output language: English
- Output format: structured PDF with citations

## Dataset Understanding To Anchor The Report

The current dataset contains three folders of `.s1p` files:

- `dry sensor`
- `wet sensor`
- `ice on sensor`

The files observed so far are consistent with a common frequency grid and a `single-port` numeric layout compatible with S11 magnitude-phase interpretation. The report should treat the current dataset as the authoritative scope and avoid speculative expansion to multi-port S-parameters.

## Report Design

The PDF should use a hybrid comparative structure:

1. Problem framing and data type
2. Dataset abstraction for `single-port S11 .s1p`
3. Representation routes
4. Shared preprocessing and normalization pipeline
5. Overfitting control and leakage prevention
6. Augmentation strategy
7. Comparative review of 10 algorithms
8. Evaluation framework for edge-oriented selection
9. Decision workflow for narrowing the model shortlist
10. Practical next steps for implementation

## Core Technical Position

The report should establish one strict common pipeline before model comparison so that algorithms are not judged under inconsistent preprocessing assumptions. It should explicitly separate:

- what is invariant across all methods
- what changes between `1D spectral` and `2D image-like` routes
- what changes by algorithm family

This prevents an invalid comparison where one model benefits from richer preprocessing than another.

## Representation Design

### Route A: 1D Spectral

Treat each sample as a frequency-ordered sequence derived from S11. The report should compare at least these candidate feature encodings:

- magnitude only
- phase only
- magnitude plus phase as dual channel
- magnitude plus unwrapped phase
- magnitude plus derived slope or local derivative features

The document should explain when phase wrapping can destabilize learning and when unwrapping or derivative encoding is preferable.

### Route B: 2D Image-Like

Treat each sample as a visual or map-like representation derived from the same S11 trace. The report should compare options such as:

- plotted grayscale curve images
- dual-channel rasterized magnitude-phase maps
- spectrogram-like or heatmap-like constructions from interpolation over the frequency axis

The report should also state the risk of introducing rendering artifacts that do not exist in the underlying measurement physics.

## Shared Preprocessing Design

The common preprocessing section should define:

- Touchstone parsing assumptions for `.s1p`
- column interpretation and unit normalization
- frequency-grid validation across files
- missing-value and malformed-file handling
- outlier inspection before training
- class-label mapping from folder structure
- deterministic sample indexing and metadata logging

The report should distinguish clearly between:

- preprocessing required to make data valid
- normalization used to help learning
- augmentation used to improve robustness

## Normalization Design

The normalization section should compare and justify:

- per-feature z-score fit on training data only
- min-max scaling fit on training data only
- robust scaling when outliers are plausible
- channel-wise normalization for multi-channel 1D inputs
- image normalization for 2D route inputs

It should explicitly reject leakage-prone practices such as fitting normalization on the full dataset before splitting.

## Overfitting And Leakage Control Design

This section should be strong and specific. The report should cover:

- stratified train-validation-test splitting
- repeated cross-validation or nested cross-validation for small datasets
- keeping preprocessing fit only on training partitions
- preventing duplicate or near-duplicate leakage across splits
- controlling capacity mismatch between model size and dataset size
- early stopping, weight decay, dropout, label smoothing, and pruning where appropriate
- confidence calibration as optional post-training analysis, not a substitute for generalization

The report should emphasize that small-sample edge sensing work is especially vulnerable to optimistic estimates from careless splits.

## Augmentation Design

The augmentation section should be physics-aware, not generic. It should compare:

- additive Gaussian noise
- frequency-local jitter or smooth perturbation
- magnitude scaling within bounded physical plausibility
- phase perturbation under bounded constraints
- frequency-axis shift only if sensor acquisition conditions justify it
- mixup-style augmentation as optional and high-risk for interpretability
- 2D image perturbations only when they preserve the underlying S11 structure

The report should explicitly warn against augmentations that create visually diverse but physically implausible samples.

## Algorithm Set

The comparative review should use these 10 methods:

1. Logistic Regression
2. Linear SVM
3. RBF SVM
4. Random Forest
5. Gradient Boosted Trees
6. Multilayer Perceptron
7. 1D CNN
8. ResNet-1D
9. Kolmogorov-Arnold Network
10. K-Means as an unsupervised structure probe, not a primary final classifier

## Algorithm Review Contract

Each algorithm subsection should answer the same questions:

- expected input shape
- compatible representation route
- required preprocessing
- preferred normalization
- augmentation compatibility
- overfitting risk under small sample size
- interpretability level
- edge deployment suitability
- expected robustness to noise
- main failure modes

The report should also state that `Keras` is a framework for implementing neural models, not a standalone algorithm, and therefore belongs in the implementation stack discussion rather than the 10-model comparison table.

## Evaluation Framework Design

The report should define an evaluation framework without filling in numerical results. It should specify:

- macro F1 as the primary balanced classification metric
- per-class recall and confusion patterns
- robustness curves across injected noise levels
- model size and inference complexity
- calibration and abstention analysis as optional later-stage checks
- representation-specific ablations
- supervised versus unsupervised interpretation boundaries

## Source Policy

The PDF should rely on primary or official sources whenever possible:

- Touchstone format specification or Keysight-compatible documentation for `.s1p` interpretation
- original or canonical papers for algorithm families
- official framework documentation only where implementation behavior matters
- no anonymous blog-style sources as core evidence

## Deliverables

The implementation phase should produce:

- one LaTeX source tree for the report
- one compiled PDF
- a references section with direct citations
- no synthetic metrics tables pretending to be experimental output

## Constraints

- No model training in this deliverable
- No benchmark numbers unless they come from cited literature and are clearly marked as external
- No claims that one model is the winner
- No extension to multi-port data
- No drift into manuscript results language

## Known Risks

- The user asked for 10 algorithms; the report must avoid becoming a shallow catalog
- The 2D route can become visually persuasive but physically weak if rasterization choices dominate
- Small-sample methodology must be treated conservatively to avoid overclaiming future generalization
- K-Means must be framed correctly as an exploratory unsupervised tool, not an equal replacement for supervised classifiers

## Implementation Notes

- The report should be authored in LaTeX and compiled to PDF
- Tables should carry the comparison burden so the prose stays readable
- A short executive summary should lead the report
- The closing section should end with a decision workflow, not a winner declaration

## Environment Note

This workspace is not currently a Git repository, so the design document cannot be committed here unless the project is moved into version control.
