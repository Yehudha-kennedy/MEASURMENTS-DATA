"""
Nested Cross-Validation
=======================
Outer loop: StratifiedGroupKFold (5 splits) – unbiased performance estimate.
Inner loop: StratifiedGroupKFold grid search – hyperparameter selection.

Group constraints are respected in **both** loops so that near-duplicate
traces never leak between train and validation.
"""
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.preprocessor import Preprocessor


# ---------------------------------------------------------------------------
# Default parameter grids
# ---------------------------------------------------------------------------
DEFAULT_GRIDS: dict = {
    "LogisticRegression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
    },
    "SVC": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.01, 0.001],
    },
}


def _param_combos(grid: dict) -> list[dict]:
    """Expand a parameter grid dict into a list of individual param dicts."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


# ---------------------------------------------------------------------------
# Inner-loop grid search
# ---------------------------------------------------------------------------
def _inner_grid_search(
    model_class,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_inner_splits: int = 3,
    random_state: int = 42,
    normalization_method: str = "z-score",
) -> tuple[dict, float]:
    """Run inner-loop grid search respecting groups.

    Parameters
    ----------
    model_class : type
        Uninstantiated sklearn estimator class.
    param_grid : dict
        Hyperparameter search space.
    X, y, groups : np.ndarray
        Training data for the *outer* fold.
    n_inner_splits : int
        Number of inner CV folds.
    random_state : int
        RNG seed.
    normalization_method : str
        Normalization to apply.

    Returns
    -------
    best_params : dict
        Parameter combination with the highest mean inner-fold macro-F1.
    best_score : float
        The corresponding mean macro-F1.
    """
    combos = _param_combos(param_grid)
    best_score = -1.0
    best_params: dict = {}

    # Check that we have enough unique groups for the inner split
    n_unique_groups = len(np.unique(groups))
    actual_inner_splits = min(n_inner_splits, n_unique_groups)
    if actual_inner_splits < 2:
        # Not enough groups to split – just return default params
        return combos[0] if combos else {}, 0.0

    inner_cv = StratifiedGroupKFold(
        n_splits=actual_inner_splits, shuffle=True, random_state=random_state
    )

    for params in combos:
        fold_scores = []
        for train_idx, val_idx in inner_cv.split(X, y, groups=groups):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            # Normalise (anti-leakage)
            preproc = Preprocessor(method=normalization_method)
            X_tr = preproc.fit_transform(X_tr)
            X_va = preproc.transform(X_va)

            # Flatten for sklearn tabular estimators
            X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
            X_va_flat = X_va.reshape(X_va.shape[0], -1)

            model = model_class(**params)
            # Ensure SVC probability mode for later ROC AUC
            if hasattr(model, "probability"):
                model.set_params(probability=True)

            model.fit(X_tr_flat, y_tr)
            preds = model.predict(X_va_flat)
            fold_scores.append(f1_score(y_va, preds, average="macro"))

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def nested_cv_evaluate(
    model_class,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str = "Model",
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    random_state: int = 42,
    normalization_method: str = "z-score",
) -> dict:
    """Run nested cross-validation with group constraints.

    Parameters
    ----------
    model_class : type
        Uninstantiated sklearn estimator.
    param_grid : dict
        Hyperparameter grid for inner search.
    X : np.ndarray, shape (N, 4001, 2)
        Raw S11 data.
    y : np.ndarray, shape (N,)
        Labels (0=dry, 1=wet, 2=ice).
    groups : np.ndarray, shape (N,)
        Group identifiers for near-duplicate control.
    model_name : str
        Human-readable label.
    n_outer_splits : int
        Outer CV folds.
    n_inner_splits : int
        Inner CV folds.
    random_state : int
        Master RNG seed.
    normalization_method : str
        Normalization applied in both loops.

    Returns
    -------
    dict
        ``best_params_per_fold``, ``outer_f1_mean``, ``outer_f1_std``,
        ``outer_acc_mean``, ``outer_acc_std``, ``outer_roc_auc_mean``,
        ``outer_roc_auc_std``.
    """
    print(f"\n=== Nested CV: {model_name} ===")
    print(f"Outer splits: {n_outer_splits}, Inner splits: {n_inner_splits}")

    outer_cv = StratifiedGroupKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=random_state
    )

    outer_f1_scores: list[float] = []
    outer_acc_scores: list[float] = []
    outer_roc_scores: list[float] = []
    best_params_per_fold: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        X_train_outer, y_train_outer = X[train_idx], y[train_idx]
        groups_train_outer = groups[train_idx]
        X_test_outer, y_test_outer = X[test_idx], y[test_idx]

        # --- inner grid search on outer-train ---
        best_params, inner_score = _inner_grid_search(
            model_class,
            param_grid,
            X_train_outer,
            y_train_outer,
            groups_train_outer,
            n_inner_splits=n_inner_splits,
            random_state=random_state + fold,
            normalization_method=normalization_method,
        )
        best_params_per_fold.append(best_params)

        # --- retrain on full outer-train with best params, evaluate on outer-test ---
        preproc = Preprocessor(method=normalization_method)
        X_tr_norm = preproc.fit_transform(X_train_outer)
        X_te_norm = preproc.transform(X_test_outer)

        X_tr_flat = X_tr_norm.reshape(X_tr_norm.shape[0], -1)
        X_te_flat = X_te_norm.reshape(X_te_norm.shape[0], -1)

        model = model_class(**best_params)
        if hasattr(model, "probability"):
            model.set_params(probability=True)

        model.fit(X_tr_flat, y_train_outer)

        preds = model.predict(X_te_flat)
        f1 = f1_score(y_test_outer, preds, average="macro")
        acc = accuracy_score(y_test_outer, preds)

        # ROC AUC
        roc = np.nan
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_te_flat)
            elif hasattr(model, "decision_function"):
                proba = model.decision_function(X_te_flat)
            else:
                proba = None

            if proba is not None and proba.ndim == 2 and proba.shape[1] == len(np.unique(y)):
                roc = roc_auc_score(
                    y_test_outer, proba, multi_class="ovr", average="macro"
                )
        except Exception:
            roc = np.nan

        outer_f1_scores.append(f1)
        outer_acc_scores.append(acc)
        outer_roc_scores.append(roc)

        print(
            f"  Fold {fold + 1}/{n_outer_splits}  |  "
            f"best_params={best_params}  |  "
            f"F1={f1:.4f}  Acc={acc:.4f}  ROC={roc:.4f}"
        )

    results = {
        "model_name": model_name,
        "best_params_per_fold": best_params_per_fold,
        "outer_f1_mean": float(np.mean(outer_f1_scores)),
        "outer_f1_std": float(np.std(outer_f1_scores)),
        "outer_acc_mean": float(np.mean(outer_acc_scores)),
        "outer_acc_std": float(np.std(outer_acc_scores)),
        "outer_roc_auc_mean": float(np.nanmean(outer_roc_scores)),
        "outer_roc_auc_std": float(np.nanstd(outer_roc_scores)),
    }

    print(f"\n[Nested CV Summary: {model_name}]")
    print(f"  Outer Macro F1 : {results['outer_f1_mean']:.4f} ± {results['outer_f1_std']:.4f}")
    print(f"  Outer Accuracy : {results['outer_acc_mean']:.4f} ± {results['outer_acc_std']:.4f}")
    print(f"  Outer ROC AUC  : {results['outer_roc_auc_mean']:.4f} ± {results['outer_roc_auc_std']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)

    # Synthetic data mimicking S11 shape
    N = 60
    X_syn = rng.standard_normal((N, 50, 2))  # smaller bins for speed
    y_syn = np.array([0] * 20 + [1] * 20 + [2] * 20)
    groups_syn = np.arange(N)  # each sample is its own group

    result = nested_cv_evaluate(
        model_class=LogisticRegression,
        param_grid={"C": [0.1, 1.0], "max_iter": [500]},
        X=X_syn,
        y=y_syn,
        groups=groups_syn,
        model_name="LR_NestedCV_Test",
        n_outer_splits=3,
        n_inner_splits=2,
    )

    assert "outer_f1_mean" in result
    assert len(result["best_params_per_fold"]) == 3
    print("\n[nested_cv.py] All self-tests passed.")
