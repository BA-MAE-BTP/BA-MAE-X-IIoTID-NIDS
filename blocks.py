"""Dependency block construction for BA-MAE: NMI-Spearman dependency matrix, hierarchical clustering, and exact-count dynamic block sampler."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import normalized_mutual_info_score

from utils import setup_logger, compute_sha256
from dataset import XIIoTIDDataModule

logger = setup_logger("bamae.blocks")


def compute_nmi_matrix(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Compute pairwise Normalized Mutual Information (NMI) across columns via discretization."""
    n_samples, n_features = X.shape
    nmi_mat = np.eye(n_features, dtype=np.float64)

    # Discretize continuous features into quantile bins
    discrete_features = np.empty((n_samples, n_features), dtype=np.int32)
    for j in range(n_features):
        col = X[:, j]
        # Robust binning with jitter to prevent duplicate edge collapsing
        jitter = 1e-6 * np.random.RandomState(42).randn(n_samples)
        try:
            discrete_features[:, j] = pd.qcut(col + jitter, q=n_bins, labels=False, duplicates="drop")
        except Exception:
            discrete_features[:, j] = pd.cut(col, bins=n_bins, labels=False)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            score = normalized_mutual_info_score(discrete_features[:, i], discrete_features[:, j])
            nmi_mat[i, j] = score
            nmi_mat[j, i] = score

    return nmi_mat


def compute_spearman_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise absolute Spearman rank correlation across columns."""
    df = pd.DataFrame(X)
    rho_mat = df.corr(method="spearman").abs().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(rho_mat, 1.0)
    return rho_mat


def compute_dependency_matrix(X_train_normal: np.ndarray, alpha: float = 0.35, n_bins: int = 10) -> np.ndarray:
    """Compute pairwise feature dependency matrix:

    D_ij = alpha * NMI_ij + (1 - alpha) * |rho_ij|

    Strictly computed on normal training data (y=0).
    """
    logger.info(f"Computing NMI matrix (alpha={alpha}) on {len(X_train_normal):,} normal training samples...")
    nmi_matrix = compute_nmi_matrix(X_train_normal, n_bins=n_bins)

    logger.info("Computing Spearman rank correlation matrix...")
    spearman_matrix = compute_spearman_matrix(X_train_normal)

    d_matrix = alpha * nmi_matrix + (1.0 - alpha) * spearman_matrix
    np.fill_diagonal(d_matrix, 1.0)
    d_matrix = np.clip(d_matrix, 0.0, 1.0)
    return d_matrix


def build_dependency_blocks(
    d_matrix: np.ndarray,
    feature_names: List[str],
    target_budget: int = 8,
    max_block_size: int = 4,
    output_csv: Path | str = "alpha_0.35_blocks_xiiotid.csv",
) -> Tuple[Dict[int, List[int]], pd.DataFrame, str]:
    """Cluster features into dependency blocks using hierarchical clustering and export mapping CSV."""
    n_features = d_matrix.shape[0]
    dist_matrix = 1.0 - d_matrix
    np.fill_diagonal(dist_matrix, 0.0)

    # Ensure strictly symmetric and non-negative
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    dist_matrix = np.clip(dist_matrix, 0.0, 1.0)
    np.fill_diagonal(dist_matrix, 0.0)

    condensed_dist = squareform(dist_matrix)
    z_linkage = linkage(condensed_dist, method="average")

    # Search for optimal number of clusters so that maximum block size <= max_block_size
    best_clusters = None
    best_k = 10
    for k in range(8, 20):
        c = fcluster(z_linkage, t=k, criterion="maxclust")
        sizes = [np.sum(c == cid) for cid in np.unique(c)]
        if max(sizes) <= max_block_size:
            best_clusters = c
            best_k = k
            break

    if best_clusters is None:
        best_clusters = fcluster(z_linkage, t=12, criterion="maxclust")

    # Map to blocks
    blocks: Dict[int, List[int]] = {}
    unique_cids = np.unique(best_clusters)
    cid_map = {cid: idx for idx, cid in enumerate(unique_cids)}

    for feat_idx, cid in enumerate(best_clusters):
        b_id = cid_map[cid]
        blocks.setdefault(b_id, []).append(feat_idx)

    rows = []
    for b_id, feats in sorted(blocks.items()):
        b_size = len(feats)
        for f_idx in feats:
            rows.append({
                "feature_idx": f_idx,
                "feature_name": feature_names[f_idx] if f_idx < len(feature_names) else f"PCA_{f_idx}",
                "block_id": b_id,
                "block_size": b_size,
            })

    df_blocks = pd.DataFrame(rows).sort_values("feature_idx").reset_index(drop=True)
    out_path = Path(output_csv)
    df_blocks.to_csv(out_path, index=False)

    sha256_hash = compute_sha256(out_path)
    logger.info(f"Exported {len(blocks)} dependency blocks to {out_path} (SHA256: {sha256_hash})")
    for b_id, feats in sorted(blocks.items()):
        logger.info(f"  Block {b_id:2d}: size={len(feats)} features={feats}")

    return blocks, df_blocks, sha256_hash


class ExactCountDynamicBlockSampler:
    """Samples whole dependency blocks such that total hidden features per row equals exactly the target budget."""

    def __init__(self, blocks: Dict[int, List[int]], n_features: int = 31) -> None:
        self.blocks = blocks
        self.n_features = n_features
        self.block_ids = sorted(blocks.keys())
        self.block_sizes = {b_id: len(feats) for b_id, feats in blocks.items()}
        self._subset_cache: Dict[int, List[List[int]]] = {}

    def _find_subsets_exact_budget(self, target_budget: int) -> List[List[int]]:
        """Find all combinations of block IDs summing to target_budget."""
        if target_budget in self._subset_cache:
            return self._subset_cache[target_budget]

        valid_subsets = []
        n_blocks = len(self.block_ids)

        def backtrack(idx: int, current_sum: int, current_blocks: List[int]):
            if current_sum == target_budget:
                valid_subsets.append(list(current_blocks))
                return
            if idx >= n_blocks or current_sum > target_budget:
                return

            b_id = self.block_ids[idx]
            b_sz = self.block_sizes[b_id]

            # Include block
            current_blocks.append(b_id)
            backtrack(idx + 1, current_sum + b_sz, current_blocks)
            current_blocks.pop()

            # Skip block
            backtrack(idx + 1, current_sum, current_blocks)

        backtrack(0, 0, [])
        self._subset_cache[target_budget] = valid_subsets
        return valid_subsets

    def sample_mask(self, batch_size: int, target_budget: int = 8, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Generate binary mask of shape (batch_size, n_features) with 1=masked, 0=visible.

        Each row has exactly target_budget elements equal to 1, formed from complete blocks.
        """
        if rng is None:
            rng = np.random.RandomState()

        subsets = self._find_subsets_exact_budget(target_budget)
        masks = np.zeros((batch_size, self.n_features), dtype=np.float32)

        if not subsets:
            # Fallback if no exact combination exists: pick blocks greedily and top off
            logger.warning(f"No exact block combinations found for budget {target_budget}; using nearest greedy blocks.")
            for row in range(batch_size):
                perm = rng.permutation(self.block_ids)
                hidden_feats = []
                for b_id in perm:
                    feats = self.blocks[b_id]
                    if len(hidden_feats) + len(feats) <= target_budget:
                        hidden_feats.extend(feats)
                # Top off if needed
                if len(hidden_feats) < target_budget:
                    remaining = [f for f in range(self.n_features) if f not in hidden_feats]
                    top_off = rng.choice(remaining, size=target_budget - len(hidden_feats), replace=False)
                    hidden_feats.extend(top_off)
                masks[row, hidden_feats] = 1.0
            return masks

        # Sample subsets uniformly
        chosen_indices = rng.randint(0, len(subsets), size=batch_size)
        for row, idx in enumerate(chosen_indices):
            block_set = subsets[idx]
            hidden_indices = []
            for b_id in block_set:
                hidden_indices.extend(self.blocks[b_id])
            masks[row, hidden_indices] = 1.0

        return masks


def sample_random_mask(batch_size: int, n_features: int = 31, target_budget: int = 8, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Generate random binary mask with exactly target_budget features hidden per row (E1 baseline)."""
    if rng is None:
        rng = np.random.RandomState()

    masks = np.zeros((batch_size, n_features), dtype=np.float32)
    for row in range(batch_size):
        hidden_indices = rng.choice(n_features, size=target_budget, replace=False)
        masks[row, hidden_indices] = 1.0
    return masks
