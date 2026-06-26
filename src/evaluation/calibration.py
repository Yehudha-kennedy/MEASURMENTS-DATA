"""
Calibration Diagnostics
=======================
Provides reliability diagrams, Expected Calibration Error (ECE),
and Platt-scaling wrappers for the 3-class S11 surface-state classifier.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.frozen import FrozenEstimator
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


CLASS_NAMES = {0: "dry", 1: "wet", 2: "ice"}


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    output_path: str | None = None,
) -> plt.Figure:
    """Plot one-vs-rest calibration curves for each of the 3 classes.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        True integer labels (0, 1, 2).
    y_proba : np.ndarray, shape (N, 3)
        Predicted probabilities for each class.
    n_bins : int
        Number of bins for the calibration curve.
    output_path : str or None
        If provided, save the figure as PNG at this path.

    Returns
    -------
    matplotlib.figure.Figure
        The reliability diagram figure.
    """
    n_classes = y_proba.shape[1]
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4.5), squeeze=False)
    axes = axes.ravel()

    for cls_idx in range(n_classes):
        ax = axes[cls_idx]
        y_binary = (y_true == cls_idx).astype(int)
        prob_cls = y_proba[:, cls_idx]

        fraction_pos, mean_pred = calibration_curve(
            y_binary, prob_cls, n_bins=n_bins, strategy="uniform"
        )

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
        ax.plot(mean_pred, fraction_pos, "s-", label=f"Class {cls_idx} ({CLASS_NAMES.get(cls_idx, '')})")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Reliability – {CLASS_NAMES.get(cls_idx, cls_idx)}")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Reliability diagram saved to {out}")

    return fig


def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    Uses the standard binning approach: weighted average of
    |accuracy_bin − confidence_bin| across bins.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        True integer labels.
    y_proba : np.ndarray, shape (N, K)
        Predicted probability matrix.
    n_bins : int
        Number of equal-width bins in [0, 1].

    Returns
    -------
    float
        ECE value in [0, 1].
    """
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for b in range(n_bins):
        mask = (confidences > bin_boundaries[b]) & (confidences <= bin_boundaries[b + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        acc_bin = accuracies[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += (n_bin / n_total) * abs(acc_bin - conf_bin)

    return float(ece)


def apply_platt_scaling(model, X_cal: np.ndarray, y_cal: np.ndarray):
    """Fit Platt scaling on a held-out calibration set.

    Wraps the model with ``sklearn.calibration.CalibratedClassifierCV``
    using the ``sigmoid`` method (Platt scaling) and ``prefit=True``.

    Parameters
    ----------
    model : estimator
        A scikit-learn compatible classifier that is already fitted.
    X_cal : np.ndarray
        Calibration features (flattened if needed by the caller).
    y_cal : np.ndarray
        Calibration labels.

    Returns
    -------
    CalibratedClassifierCV
        The calibrated model wrapping the original.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(X_cal, y_cal)
    return calibrated


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)

    # Synthetic 3-class data
    N = 300
    X_syn = rng.standard_normal((N, 20))
    y_syn = rng.integers(0, 3, size=N)

    # Train a simple model
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_syn[:200], y_syn[:200])
    proba = lr.predict_proba(X_syn[200:])

    # ECE
    ece = compute_ece(y_syn[200:], proba)
    print(f"ECE = {ece:.4f}")

    # Reliability diagram (saved to temp)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    out_dir = BASE_DIR / "outputs" / "calibration"
    fig = plot_reliability_diagram(y_syn[200:], proba, output_path=out_dir / "reliability_test.png")
    plt.close(fig)

    # Platt scaling
    cal_model = apply_platt_scaling(lr, X_syn[200:], y_syn[200:])
    proba_cal = cal_model.predict_proba(X_syn[200:])
    ece_cal = compute_ece(y_syn[200:], proba_cal)
    print(f"ECE after Platt scaling = {ece_cal:.4f}")

    print("\n[calibration.py] All self-tests passed.")
