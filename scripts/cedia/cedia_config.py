"""
CEDIA HPC Configuration — Mohamed S11 Edge Classification Project
==================================================================
Single source of truth for all CEDIA launcher scripts.
Updated for Mateo's current machine (MATEO PC, June 2026).
"""
import os
from pathlib import Path

# ─── SSH Connection ──────────────────────────────────────────────────────────
HOSTNAME = "hpc.cedia.edu.ec"
USERNAME = "kevin.landazuri__yachaytech.edu.ec"
KEY_FILE = os.path.expanduser("~/.ssh/cedia_rsa")

# ─── Remote Paths (CEDIA HPC) ───────────────────────────────────────────────
REMOTE_HOME = f"/home/{USERNAME}"
REMOTE_BASE = f"{REMOTE_HOME}/Mateo Gavilanes/Mohamed"
REMOTE_DIRS = {
    "src":     f"{REMOTE_BASE}/src",
    "data":    f"{REMOTE_BASE}/data",
    "configs": f"{REMOTE_BASE}/configs",
    "scripts": f"{REMOTE_BASE}/scripts",
    "outputs": f"{REMOTE_BASE}/outputs",
    "models":  f"{REMOTE_BASE}/models",
}

# ─── Local Paths (This Machine) ─────────────────────────────────────────────
LOCAL_PROJECT_ROOT = Path(r"c:\Users\MATEO\Desktop\PERSONAL GROWTH\PROJECTS\Mohamed")
LOCAL_SRC = LOCAL_PROJECT_ROOT / "src"
LOCAL_DATA_RAW = LOCAL_PROJECT_ROOT / "data_exp (1)" / "data_exp"
LOCAL_DATA_PROCESSED = LOCAL_PROJECT_ROOT / "data" / "processed"
LOCAL_CONFIGS = LOCAL_PROJECT_ROOT / "configs"
LOCAL_SCRIPTS = LOCAL_PROJECT_ROOT / "scripts" / "cedia"
LOCAL_DOCS = LOCAL_PROJECT_ROOT / "docs"

# ─── SLURM Settings ─────────────────────────────────────────────────────────
SLURM_PARTITION = "gpu"           # adjust if CEDIA has different partitions
SLURM_NODES = 1
SLURM_GPUS = 1
SLURM_CPUS_PER_TASK = 4
SLURM_MEM = "16G"
SLURM_TIME = "04:00:00"          # 4 hours max for training jobs
SLURM_JOB_NAME_PREFIX = "s11_"

# ─── File Filters ────────────────────────────────────────────────────────────
UPLOAD_EXTENSIONS = {".py", ".sh", ".yaml", ".yml", ".json", ".csv", ".md", ".txt"}
SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".git", "__MACOSX", ".ipynb_checkpoints"}

# ─── Logging Patterns ───────────────────────────────────────────────────────
LOG_PATTERN = "s11_*_log_*.out"
ERR_PATTERN = "s11_*_err_*.err"
