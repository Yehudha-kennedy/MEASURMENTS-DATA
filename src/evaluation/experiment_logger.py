"""
Experiment Logger
=================
Records every evaluation run with full reproducibility metadata:
environment versions, random seeds, normalization stats, class
distribution, model hyperparameters, and aggregate results.
Persists as structured JSON.
"""
import json
import sys
import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _safe_version(module_name: str) -> str:
    """Return the version string of *module_name*, or 'not installed'."""
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not installed"


class ExperimentLogger:
    """Accumulates run records and writes them to a single JSON file.

    Attributes
    ----------
    log : dict
        Top-level log structure with ``environment``, ``runs`` list, and
        ``created_at`` timestamp.
    """

    def __init__(self, random_seed: int = 42):
        """Initialise the logger and snapshot the current environment.

        Parameters
        ----------
        random_seed : int
            The master random seed used for the experiment session.
        """
        self.log: dict[str, Any] = {
            "created_at": datetime.datetime.now().isoformat(),
            "environment": {
                "python_version": sys.version,
                "numpy": _safe_version("numpy"),
                "sklearn": _safe_version("sklearn"),
                "torch": _safe_version("torch"),
                "xgboost": _safe_version("xgboost"),
                "scipy": _safe_version("scipy"),
                "pandas": _safe_version("pandas"),
            },
            "random_seed": random_seed,
            "runs": [],
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_class_distribution(self, y: np.ndarray) -> None:
        """Record the number of samples per class.

        Parameters
        ----------
        y : np.ndarray
            Label array (integer-encoded).
        """
        unique, counts = np.unique(y, return_counts=True)
        self.log["class_distribution"] = {
            int(k): int(v) for k, v in zip(unique, counts)
        }

    def set_normalization_stats(
        self,
        method: str,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        vmin: np.ndarray | None = None,
        vmax: np.ndarray | None = None,
    ) -> None:
        """Record normalization method and summary statistics.

        Only scalar summaries (global mean-of-mean, etc.) are stored
        to keep the JSON compact.

        Parameters
        ----------
        method : str
            E.g. ``'z-score'`` or ``'min-max'``.
        mean, std, vmin, vmax : np.ndarray or None
            Arrays of per-feature statistics.  Scalar summaries are
            extracted automatically.
        """
        stats: dict[str, Any] = {"method": method}
        if mean is not None:
            stats["mean_of_means"] = float(np.mean(mean))
        if std is not None:
            stats["mean_of_stds"] = float(np.mean(std))
        if vmin is not None:
            stats["global_min"] = float(np.min(vmin))
        if vmax is not None:
            stats["global_max"] = float(np.max(vmax))
        self.log["normalization"] = stats

    def log_run(
        self,
        model_name: str,
        params: dict,
        results: dict,
        norm_method: str = "z-score",
    ) -> None:
        """Append a single evaluation run to the log.

        Parameters
        ----------
        model_name : str
            Human-readable model identifier.
        params : dict
            Hyperparameters used for this run.
        results : dict
            Metrics dictionary (F1, Accuracy, ROC AUC, etc.).
        norm_method : str
            Normalization method used during this run.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_name": model_name,
            "hyperparameters": _jsonable(params),
            "normalization_method": norm_method,
            "results": _jsonable(results),
        }
        self.log["runs"].append(entry)

    def save(self, output_path: str | Path) -> None:
        """Persist the full log as pretty-printed JSON.

        Parameters
        ----------
        output_path : str or Path
            Destination file path (e.g. ``outputs/experiment_log.json``).
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2, default=str)
        print(f"Experiment log saved to {out}")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_runs = len(self.log["runs"])
        return f"<ExperimentLogger runs={n_runs}>"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger = ExperimentLogger(random_seed=42)

    # Class distribution
    y_dummy = np.array([0] * 30 + [1] * 50 + [2] * 20)
    logger.set_class_distribution(y_dummy)

    # Normalization stats
    logger.set_normalization_stats(
        method="z-score",
        mean=np.zeros((4001, 2)),
        std=np.ones((4001, 2)),
    )

    # Log two dummy runs
    logger.log_run(
        model_name="LogisticRegression",
        params={"C": 1.0, "max_iter": 1000},
        results={"Val_F1_mean": 0.85, "Val_Acc_mean": 0.88},
        norm_method="z-score",
    )
    logger.log_run(
        model_name="RandomForest",
        params={"n_estimators": 100, "max_depth": 10},
        results={"Val_F1_mean": 0.90, "Val_Acc_mean": 0.91},
        norm_method="z-score",
    )

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    out_file = BASE_DIR / "outputs" / "experiment_log_test.json"
    logger.save(out_file)

    # Verify round-trip
    with open(out_file) as f:
        loaded = json.load(f)
    assert len(loaded["runs"]) == 2, "Expected 2 runs"
    assert loaded["class_distribution"]["0"] == 30
    print(f"\n[experiment_logger.py] All self-tests passed.  {logger}")
