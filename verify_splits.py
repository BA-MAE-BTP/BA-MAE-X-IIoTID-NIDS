"""Verification script for frozen data splits and SHA256 hash integrity."""

from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np

from utils import setup_logger, compute_sha256
from dataset import XIIoTIDDataModule

logger = setup_logger("bamae.verify_splits")


def run_verification(handoff_dir: str = r"C:\Users\Student\Downloads\safe_main") -> bool:
    """Load splits, verify counts, and compare SHA256 against splits_hash_log.json."""
    logger.info("=" * 70)
    logger.info("VERIFYING HANDOFF SPLIT INTEGRITY AGAINST SHA256 LOG")
    logger.info("=" * 70)

    data_module = XIIoTIDDataModule(handoff_dir=handoff_dir)
    train_idx, val_idx, test_idx = data_module.get_or_create_splits()

    logger.info(f"Loaded train_idx: shape {train_idx.shape}, dtype {train_idx.dtype}")
    logger.info(f"Loaded val_idx:   shape {val_idx.shape}, dtype {val_idx.dtype}")
    logger.info(f"Loaded test_idx:  shape {test_idx.shape}, dtype {test_idx.dtype}")

    # Check for zero overlap across splits
    set_tr = set(train_idx)
    set_val = set(val_idx)
    set_te = set(test_idx)

    assert len(set_tr.intersection(set_val)) == 0, "DATA LEAKAGE: train and val overlap!"
    assert len(set_tr.intersection(set_te)) == 0, "DATA LEAKAGE: train and test overlap!"
    assert len(set_val.intersection(set_te)) == 0, "DATA LEAKAGE: val and test overlap!"
    logger.info("[PASSED] Zero index overlap confirmed across train, val, and test splits.")

    # Verify SHA256
    results = data_module.verify_split_integrity()
    logger.info("-" * 70)
    for split_name in ["train_idx", "val_idx", "test_idx"]:
        info = results[split_name]
        status = "MATCH [OK]" if info["match"] else "MISMATCH [FAIL]"
        logger.info(f"  {split_name:10s}: {status}")
        logger.info(f"    Expected: {info['expected']}")
        logger.info(f"    Actual  : {info['actual']}")
    logger.info("-" * 70)

    if results["all_passed"]:
        logger.info("[SUCCESS] 100% SHA256 statistical alignment and integrity verified!")
        return True
    else:
        logger.error("[FAILURE] SHA256 hash mismatch detected!")
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
