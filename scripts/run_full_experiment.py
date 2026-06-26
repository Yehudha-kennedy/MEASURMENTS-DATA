"""
Master Experiment Runner
========================
Executes the full 3xN experiment matrix:
- 3 Normalization Methods (z-score, min-max, robust)
- × Classical Models (Logistic Regression, Random Forest, XGBoost, RBF SVM)
- × Deep Models (MLP, ResNet-1D, KAN)

Logs all results to a structured JSON using ExperimentLogger.
"""
import argparse
import numpy as np
from pathlib import Path
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from evaluation.cv_orchestrator import CVOrchestrator
from evaluation.experiment_logger import ExperimentLogger
from route_a_1d_spectral.models.deep_models import MLPWrapper, ResNet1DWrapper, KANWrapper

def main(data_dir, output_dir, n_splits=5, n_repeats=3, test_run=False):
    data_path = Path(data_dir) / "dataset_grouped.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run preprocessor.py first.")
        
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    X, y, groups = data['X'], data['y'], data['groups']
    
    if test_run:
        print("\n--- TEST RUN MODE ---")
        # Subsample to speed up tests drastically but keep all classes
        idx = np.concatenate([np.where(y==0)[0][:20], np.where(y==1)[0][:20], np.where(y==2)[0][:20]])
        X, y, groups = X[idx], y[idx], groups[idx]
        n_splits, n_repeats = 2, 1
    
    logger = ExperimentLogger(random_seed=42)
    logger.set_class_distribution(y)
    
    # 3xN Matrix parameters
    norm_methods = ["z-score", "min-max", "robust"]
    
    tabular_models = [
        ("LogisticRegression", LogisticRegression, {'max_iter': 1000, 'random_state': 42}),
        ("RandomForest", RandomForestClassifier, {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}),
        ("XGBoost", XGBClassifier, {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'n_jobs': -1, 'eval_metric': 'mlogloss'}),
        ("RBF_SVM", SVC, {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42})
    ]
    
    deep_models = [
        ("MLP", MLPWrapper, {"epochs": 2 if test_run else 30, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4, "device": "cpu"}),
        ("ResNet1D", ResNet1DWrapper, {"epochs": 2 if test_run else 30, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4, "device": "cpu"}),
        ("KAN", KANWrapper, {"epochs": 2 if test_run else 30, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4, "grid_size": 5, "device": "cpu"})
    ]
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for norm in norm_methods:
        print(f"\n=======================================================")
        print(f" EXPERIMENT BLOCK: Normalization = {norm.upper()}")
        print(f"=======================================================")
        
        orch = CVOrchestrator(n_splits=n_splits, n_repeats=n_repeats, use_augmentation=False, random_state=42)
        # Hack to override the normalizer via constructor initialization since its a hardcoded parameter currently.
        # Let's fix this properly. Wait, cv_orchestrator __init__ was upgraded by the subagent to take normalization_method!
        orch.normalization_method = norm
        
        # 1. Tabular Models
        for name, cls, kwargs in tabular_models:
            print(f"\nEvaluating {name} with {norm}...")
            try:
                res = orch.evaluate(cls, kwargs, X, y, groups, model_name=name)
                logger.log_run(model_name=name, params=kwargs, results=res, norm_method=norm)
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                
        # 2. Deep Models
        for name, cls, kwargs in deep_models:
            print(f"\nEvaluating {name} with {norm}...")
            try:
                res = orch.evaluate(cls, kwargs, X, y, groups, model_name=name)
                logger.log_run(model_name=name, params=kwargs, results=res, norm_method=norm)
            except Exception as e:
                print(f"Error evaluating {name}: {e}")

    # Save Master Log
    json_path = out_path / "master_experiment_log.json"
    logger.save(json_path)
    print(f"\n[DONE] All experiments complete. Master log saved to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full 3xN experiment matrix.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory with dataset_grouped.npz")
    parser.add_argument("--out_dir", type=str, default="outputs/experiments", help="Output directory for JSON log")
    parser.add_argument("--splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--repeats", type=int, default=3, help="Number of CV repeats")
    parser.add_argument("--test_run", action="store_true", help="Run a tiny subset for testing")
    
    args = parser.parse_args()
    
    base = Path(__file__).resolve().parent.parent
    data_dir = base / args.data_dir
    out_dir = base / args.out_dir
    
    main(data_dir, out_dir, n_splits=args.splits, n_repeats=args.repeats, test_run=args.test_run)
