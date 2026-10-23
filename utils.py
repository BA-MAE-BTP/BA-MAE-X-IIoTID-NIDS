"""Utility functions for BA-MAE: seeding, SHA256 integrity, stratification, and statistics."""

from __future__ import annotations
import hashlib
import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import scipy.stats as stats


def setup_logger(name: str = "bamae", log_file: Optional[Path | str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure structured logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


def set_seed(seed: int = 42) -> None:
    """Set global random seed across Python, NumPy, and PyTorch for 100% determinism."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def compute_sha256(filepath: Path | str) -> str:
    """Compute hex SHA256 hash of a file on disk."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    sha256 = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_array_sha256(arr: np.ndarray) -> str:
    """Compute SHA256 of array data buffer directly."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def assign_flow_volume_strata(flow_volumes: np.ndarray) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
    """Stratify network flow traffic into 5 volume categories based on quantiles.

    Strata:
    - 0: Body      (0.00 to 0.50 quantile)
    - 1: Light     (0.50 to 0.75 quantile)
    - 2: Moderate  (0.75 to 0.90 quantile)
    - 3: Heavy     (0.90 to 0.99 quantile)
    - 4: Extreme   (0.99 to 1.00 quantile)

    Returns:
        strata_labels: ndarray of ints (0 to 4)
        thresholds: dictionary of stratum range definitions
    """
    volumes = np.nan_to_num(flow_volumes, nan=0.0, posinf=0.0, neginf=0.0)
    q50 = float(np.percentile(volumes, 50.0))
    q75 = float(np.percentile(volumes, 75.0))
    q90 = float(np.percentile(volumes, 90.0))
    q99 = float(np.percentile(volumes, 99.0))

    strata = np.zeros(len(volumes), dtype=np.int64)
    strata[volumes > q50] = 1
    strata[volumes > q75] = 2
    strata[volumes > q90] = 3
    strata[volumes > q99] = 4

    thresholds = {
        "Body (0-50%)": (float(volumes.min()), q50),
        "Light (50-75%)": (q50, q75),
        "Moderate (75-90%)": (q75, q90),
        "Heavy (90-99%)": (q90, q99),
        "Extreme (>99%)": (q99, float(volumes.max())),
    }
    return strata, thresholds


def compute_mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, Any]:
    """Execute McNemar's paired test of statistical significance between two model predictions.

    Contingency table:
      b: Model A correct, Model B wrong
      c: Model A wrong, Model B correct

    Statistic with Edward's continuity correction:
      chi2 = (|b - c| - 1)^2 / (b + c)
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & (~correct_b)))  # A correct, B wrong
    c = int(np.sum((~correct_a) & correct_b))  # A wrong, B correct
    a_agree_correct = int(np.sum(correct_a & correct_b))
    d_agree_wrong = int(np.sum((~correct_a) & (~correct_b)))

    total_discordant = b + c
    if total_discordant == 0:
        stat = 0.0
        p_value = 1.0
    else:
        stat = float((abs(b - c) - 1.0) ** 2 / total_discordant)
        p_value = float(stats.chi2.sf(stat, df=1))

    return {
        "statistic": stat,
        "p_value": p_value,
        "b_model_a_superior": b,
        "c_model_b_superior": c,
        "both_correct": a_agree_correct,
        "both_wrong": d_agree_wrong,
        "significant_at_05": bool(p_value < 0.05),
        "significant_at_01": bool(p_value < 0.01),
    }


def compute_bootstrap_ci(
    diff_scores: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute paired non-parametric bootstrap confidence interval for metric differences.

    Returns:
        (mean_diff, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n = len(diff_scores)
    means = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        boot_sample = rng.choice(diff_scores, size=n, replace=True)
        means[i] = np.mean(boot_sample)

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1.0 - alpha) * 100))
    mean_val = float(np.mean(diff_scores))
    return mean_val, lower, upper
