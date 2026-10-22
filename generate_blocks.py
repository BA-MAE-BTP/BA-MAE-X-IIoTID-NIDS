"""Runner script to generate alpha_0.35_blocks_xiiotid.csv from normal training traffic."""

import sys
from pathlib import Path
import numpy as np

from utils import setup_logger, compute_sha256
from dataset import XIIoTIDDataModule
from blocks import compute_dependency_matrix, build_dependency_blocks

logger = setup_logger("bamae.generate_blocks")


def main():
    logger.info("Initializing XIIoTID DataModule...")
    dm = XIIoTIDDataModule()
    X_train_pca, _, _ = dm.fit_and_extract_pca_features()
    y_train = dm.y_raw[dm.train_idx]

    normal_mask = (y_train == 0)
    X_train_normal = X_train_pca[normal_mask]
    logger.info(f"Extracting dependency scores on {len(X_train_normal):,} normal training rows...")

    # For fast mutual information computation while maintaining high statistical fidelity,
    # sample 50,000 normal rows if dataset is large, or use full
    n_sample = min(50000, len(X_train_normal))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_train_normal), size=n_sample, replace=False)
    X_sub = X_train_normal[sample_idx]

    d_matrix = compute_dependency_matrix(X_sub, alpha=0.35, n_bins=10)
    blocks, df_blocks, sha256_hash = build_dependency_blocks(
        d_matrix=d_matrix,
        feature_names=dm.selected_feature_names,
        target_budget=8,
        output_csv="alpha_0.35_blocks_xiiotid.csv",
    )

    logger.info(f"Successfully generated alpha_0.35_blocks_xiiotid.csv")
    logger.info(f"SHA256: {sha256_hash}")


if __name__ == "__main__":
    main()
