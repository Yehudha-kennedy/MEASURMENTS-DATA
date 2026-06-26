"""
Cross-Validation Orchestrator
=============================
Implements Repeated Stratified Group K-Fold CV.
Ensures Preprocessing and Augmentation are strictly confined to the training folds.

Returns extended metrics: per-class F1, accuracy, ROC AUC, and an
aggregated confusion matrix across all folds.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.preprocessor import Preprocessor
from data.augmentor import S11Augmentor

CLASS_NAMES = ["dry", "wet", "ice"]


class CVOrchestrator:
    """Repeated Stratified Group K-Fold evaluator for S11 classifiers.

    Parameters
    ----------
    n_splits : int
        Number of CV folds per repeat.
    n_repeats : int
        Number of repeats (each with a different random split).
    random_state : int
        Master random seed.
    use_augmentation : bool
        Whether to apply physics-aware augmentation to training folds.
    normalization_method : str
        Normalization method forwarded to ``Preprocessor``
        (default ``'z-score'``).
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 5,
        random_state: int = 42,
        use_augmentation: bool = False,
        normalization_method: str = "z-score",
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.use_augmentation = use_augmentation
        self.normalization_method = normalization_method

    def evaluate(
        self,
        model_class,
        model_kwargs: dict,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        model_name: str = "Model",
        output_dir: str | Path | None = None,
    ) -> dict:
        """Run repeated stratified group k-fold cross-validation.

        Parameters
        ----------
        model_class : type
            Uninstantiated sklearn-compatible model class.
        model_kwargs : dict
            Keyword arguments forwarded to ``model_class()``.
        X : np.ndarray, shape (N, 4001, 2)
            Raw S11 data.
        y : np.ndarray, shape (N,)
            Integer labels (0=dry, 1=wet, 2=ice).
        groups : np.ndarray, shape (N,)
            Group identifiers for near-duplicate control.
        model_name : str
            Human-readable model identifier.
        output_dir : str, Path, or None
            If provided, the aggregated confusion matrix is saved here.

        Returns
        -------
        dict
            Metrics dictionary containing Train/Val F1, Accuracy,
            ROC AUC (mean ± std), per-class F1, and model name.
        """
        print(f"\n--- Starting Repeated CV for {model_name} ---")
        print(
            f"Splits: {self.n_splits}, Repeats: {self.n_repeats}, "
            f"Augmentation: {self.use_augmentation}, "
            f"Normalization: {self.normalization_method}"
        )

        # Pre-initialise accumulators (bug fix: no more fragile `locals()` check)
        all_train_f1: list[float] = []
        all_val_f1: list[float] = []
        all_val_acc: list[float] = []
        all_val_roc: list[float] = []
        all_per_class_f1: list[np.ndarray] = []

        n_classes = len(np.unique(y))
        aggregated_cm = np.zeros((n_classes, n_classes), dtype=int)

        # Scikit-learn doesn't have RepeatedStratifiedGroupKFold,
        # so we manually repeat with different random seeds.
        for rep in range(self.n_repeats):
            cv = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state + rep,
            )

            for fold, (train_idx, val_idx) in enumerate(
                cv.split(X, y, groups=groups)
            ):
                X_train_raw, y_train = X[train_idx], y[train_idx]
                X_val_raw, y_val = X[val_idx], y[val_idx]

                # 1. Normalization (fit ONLY on train — anti-leakage)
                preproc = Preprocessor(method=self.normalization_method)
                X_train_norm = preproc.fit_transform(X_train_raw)
                X_val_norm = preproc.transform(X_val_raw)

                # 2. Augmentation (ONLY on train, AFTER normalization)
                if self.use_augmentation:
                    aug = S11Augmentor()
                    X_train_norm, y_train = aug.augment_dataset(
                        X_train_norm, y_train, k_factor=2
                    )

                # 3. Model Training
                model = model_class(**model_kwargs)

                # Flatten for tabular sklearn models
                if hasattr(model, "fit") and "sklearn" in model.__module__:
                    X_train_fit = X_train_norm.reshape(X_train_norm.shape[0], -1)
                    X_val_eval = X_val_norm.reshape(X_val_norm.shape[0], -1)
                else:
                    X_train_fit, X_val_eval = X_train_norm, X_val_norm

                model.fit(X_train_fit, y_train)

                # 4. Evaluation
                train_pred = model.predict(X_train_fit)
                val_pred = model.predict(X_val_eval)

                # Probability / decision-function estimates for ROC AUC
                val_proba = None
                if hasattr(model, "predict_proba"):
                    val_proba = model.predict_proba(X_val_eval)
                elif hasattr(model, "decision_function"):
                    val_proba = model.decision_function(X_val_eval)

                train_f1 = f1_score(y_train, train_pred, average="macro")
                val_f1 = f1_score(y_val, val_pred, average="macro")
                val_acc = accuracy_score(y_val, val_pred)

                # Per-class F1: returns array [f1_class0, f1_class1, f1_class2]
                per_class = f1_score(y_val, val_pred, average=None)
                all_per_class_f1.append(per_class)

                # ROC AUC (multi-class, OVR)
                try:
                    if val_proba is not None and val_proba.ndim == 2:
                        if val_proba.shape[1] == n_classes:
                            val_roc_auc = roc_auc_score(
                                y_val,
                                val_proba,
                                multi_class="ovr",
                                average="macro",
                            )
                        else:
                            val_roc_auc = np.nan
                    else:
                        val_roc_auc = np.nan
                except Exception:
                    val_roc_auc = np.nan

                # Confusion matrix (accumulate across folds)
                cm_fold = confusion_matrix(
                    y_val, val_pred, labels=list(range(n_classes))
                )
                aggregated_cm += cm_fold

                all_train_f1.append(train_f1)
                all_val_f1.append(val_f1)
                all_val_acc.append(val_acc)
                all_val_roc.append(val_roc_auc)

        # ------------------------------------------------------------------
        # Aggregate results
        # ------------------------------------------------------------------
        mean_train_f1 = float(np.mean(all_train_f1))
        std_train_f1 = float(np.std(all_train_f1))
        mean_val_f1 = float(np.mean(all_val_f1))
        std_val_f1 = float(np.std(all_val_f1))
        mean_val_acc = float(np.mean(all_val_acc))
        std_val_acc = float(np.std(all_val_acc))
        mean_val_roc = float(np.nanmean(all_val_roc))
        std_val_roc = float(np.nanstd(all_val_roc))

        # Per-class F1 (mean across all folds)
        stacked = np.array(all_per_class_f1)  # shape (n_folds_total, n_classes)
        per_class_f1_dict = {}
        for cls_idx, name in enumerate(CLASS_NAMES[:n_classes]):
            per_class_f1_dict[name] = float(np.mean(stacked[:, cls_idx]))

        # ------------------------------------------------------------------
        # Print summary
        # ------------------------------------------------------------------
        print(f"\n[CV Summary: {model_name}]")
        print(f"Train Macro F1: {mean_train_f1:.4f} ± {std_train_f1:.4f}")
        print(f"Val Macro F1:   {mean_val_f1:.4f} ± {std_val_f1:.4f}")
        print(f"Val Accuracy:   {mean_val_acc:.4f} ± {std_val_acc:.4f}")
        if not np.isnan(mean_val_roc):
            print(f"Val ROC AUC:    {mean_val_roc:.4f} ± {std_val_roc:.4f} (OVR Macro)")
        print(f"Per-Class F1:   {per_class_f1_dict}")

        # ------------------------------------------------------------------
        # Save aggregated confusion matrix
        # ------------------------------------------------------------------
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=aggregated_cm,
                display_labels=CLASS_NAMES[:n_classes],
            )
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            ax.set_title(f"Aggregated Confusion Matrix – {model_name}")
            fig.tight_layout()
            cm_path = out / f"{model_name}_confusion_matrix.png"
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            print(f"Confusion matrix saved to {cm_path}")

        return {
            "Model": model_name,
            "Train_F1_mean": mean_train_f1,
            "Train_F1_std": std_train_f1,
            "Val_F1_mean": mean_val_f1,
            "Val_F1_std": std_val_f1,
            "Val_Acc_mean": mean_val_acc,
            "Val_Acc_std": std_val_acc,
            "Val_ROC_AUC_mean": mean_val_roc,
            "Val_ROC_AUC_std": std_val_roc,
            "Per_Class_F1": per_class_f1_dict,
        }


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    data_path = BASE_DIR / "data" / "processed" / "dataset_grouped.npz"

    if data_path.exists():
        data = np.load(data_path)
        X, y, groups = data["X"], data["y"], data["groups"]

        orch = CVOrchestrator(
            n_splits=5,
            n_repeats=3,
            use_augmentation=False,
            normalization_method="z-score",
        )
        results = orch.evaluate(
            model_class=LogisticRegression,
            model_kwargs={"max_iter": 1000, "random_state": 42},
            X=X,
            y=y,
            groups=groups,
            model_name="Logistic_Regression_Baseline",
            output_dir=BASE_DIR / "outputs" / "cv_results",
        )
        print(f"\nReturned keys: {list(results.keys())}")
    else:
        print(f"Dataset not found at {data_path}")
        print("Running synthetic self-test instead...")

        rng = np.random.default_rng(42)
        N = 60
        X_syn = rng.standard_normal((N, 50, 2))
        y_syn = np.array([0] * 20 + [1] * 20 + [2] * 20)
        groups_syn = np.arange(N)

        orch = CVOrchestrator(
            n_splits=3,
            n_repeats=2,
            use_augmentation=False,
            normalization_method="z-score",
        )
        results = orch.evaluate(
            model_class=LogisticRegression,
            model_kwargs={"max_iter": 500, "random_state": 42},
            X=X_syn,
            y=y_syn,
            groups=groups_syn,
            model_name="LR_SyntheticTest",
            output_dir=BASE_DIR / "outputs" / "cv_results",
        )
        print(f"\nReturned keys: {list(results.keys())}")
        assert "Val_Acc_mean" in results, "Missing Val_Acc_mean"
        assert "Val_ROC_AUC_mean" in results, "Missing Val_ROC_AUC_mean"
        assert "Per_Class_F1" in results, "Missing Per_Class_F1"
        assert set(results["Per_Class_F1"].keys()) == {"dry", "wet", "ice"}
        print("[cv_orchestrator.py] All self-tests passed.")
