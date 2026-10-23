"""Dataset module for BA-MAE: handoff integration, frozen splits, PCA extraction, and PyTorch loaders."""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except (ImportError, OSError):
    torch = None
    Dataset = object
    DataLoader = None

from utils import setup_logger, compute_sha256, assign_flow_volume_strata

logger = setup_logger("bamae.dataset")

# Ensure handoff directory is available on sys.path
SAFE_HANDOFF_PATHS = [
    Path(r"C:\Users\Student\Downloads\safe_main\safe_main"),
    Path(r"C:\Users\Student\Downloads\safe_main"),
]
for p in SAFE_HANDOFF_PATHS:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from safe_ids.data.x_iiotid import XIIoTIDLoader
    from safe_ids.preprocessing.pca_selector import PCAFeatureSelector
except ImportError as err:
    logger.warning(f"Could not import SAFE modules from handoff path: {err}")
    XIIoTIDLoader = None
    PCAFeatureSelector = None


class TabularDataset(Dataset):
    """PyTorch Dataset for 31-dimensional tabular representations."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        flow_volumes: Optional[np.ndarray] = None,
        strata: Optional[np.ndarray] = None,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

        if flow_volumes is not None:
            self.flow_volumes = torch.as_tensor(flow_volumes, dtype=torch.float32)
        else:
            self.flow_volumes = torch.zeros(len(features), dtype=torch.float32)

        if strata is not None:
            self.strata = torch.as_tensor(strata, dtype=torch.long)
        else:
            self.strata = torch.zeros(len(features), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx], self.flow_volumes[idx], self.strata[idx]


class XIIoTIDDataModule:
    """Manages raw ingestion, frozen split indexing, SHA256 integrity, and PCA feature transformation."""

    def __init__(
        self,
        raw_csv_path: Path | str = r"C:\Bamae_safedataset\X-IIoTID dataset.csv",
        handoff_dir: Path | str = r"C:\Users\Student\Downloads\safe_main",
        k_pca: int = 31,
        pca_variance_threshold: float = 0.95,
        random_seed: int = 42,
    ) -> None:
        self.raw_csv_path = Path(raw_csv_path)
        self.handoff_dir = Path(handoff_dir)
        self.k_pca = k_pca
        self.pca_variance_threshold = pca_variance_threshold
        self.random_seed = random_seed

        # Internal arrays
        self.X_raw: Optional[np.ndarray] = None
        self.y_raw: Optional[np.ndarray] = None
        self.raw_feature_names: List[str] = []
        self.flow_volumes_raw: Optional[np.ndarray] = None

        # Split indices
        self.train_idx: Optional[np.ndarray] = None
        self.val_idx: Optional[np.ndarray] = None
        self.test_idx: Optional[np.ndarray] = None

        # Transformed PCA features
        self.pca_selector: Optional[Any] = None
        self.X_train_pca: Optional[np.ndarray] = None
        self.X_val_pca: Optional[np.ndarray] = None
        self.X_test_pca: Optional[np.ndarray] = None
        self.selected_feature_names: List[str] = []

        # Strata
        self.strata_train: Optional[np.ndarray] = None
        self.strata_val: Optional[np.ndarray] = None
        self.strata_test: Optional[np.ndarray] = None
        self.strata_thresholds: Dict[str, Tuple[float, float]] = {}

    def load_raw_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ingest raw dataset using SAFE handoff logic or native fallback."""
        logger.info(f"Loading raw dataset from {self.raw_csv_path}...")
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"Raw dataset CSV does not exist at {self.raw_csv_path}")

        if XIIoTIDLoader is not None:
            loader = XIIoTIDLoader(csv_path=self.raw_csv_path, zip_path=None)
            features_df, labels_series, feature_names = loader.load_raw()
        else:
            logger.info("Using embedded ingestion fallback matching XIIoTIDLoader schema...")
            df = pd.read_csv(self.raw_csv_path, low_memory=False)
            labels_series = df["class3"].astype(str).str.strip().apply(lambda x: 0 if x.lower() == "normal" else 1)
            cols_to_drop = [c for c in ["Date", "Timestamp", "Scr_IP", "Scr_port", "Des_IP", "Des_port", "class1", "class2", "class3"] if c in df.columns]
            features_df = df.drop(columns=cols_to_drop).copy()
            bool_replacements = {"TRUE": 1.0, "True": 1.0, "true": 1.0, "FALSE": 0.0, "False": 0.0, "false": 0.0}
            for col in features_df.columns:
                if not pd.api.types.is_numeric_dtype(features_df[col]):
                    s = features_df[col].astype(str).str.strip().replace(bool_replacements)
                    converted = pd.to_numeric(s, errors="coerce")
                    if converted.notna().sum() > 0.8 * len(converted):
                        features_df[col] = converted.fillna(0.0).astype(np.float32)
                    else:
                        features_df[col] = pd.factorize(features_df[col])[0].astype(np.float32)
            features_df = features_df.astype(np.float32)
            feature_names = list(features_df.columns)

        # Extract flow volume indicator for stratification (total_bytes or total_packet)
        raw_df_bytes = None
        if "total_bytes" in features_df.columns:
            raw_df_bytes = features_df["total_bytes"].values.astype(np.float32)
        elif "total_packet" in features_df.columns:
            raw_df_bytes = features_df["total_packet"].values.astype(np.float32)
        else:
            raw_df_bytes = np.ones(len(features_df), dtype=np.float32)

        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.X_raw = features_df.values.astype(np.float32)
        self.y_raw = labels_series.values.astype(np.int64)
        self.raw_feature_names = feature_names
        self.flow_volumes_raw = raw_df_bytes

        logger.info(f"Loaded {len(self.X_raw)} rows with {self.X_raw.shape[1]} raw input features.")
        return self.X_raw, self.y_raw, self.flow_volumes_raw

    def get_or_create_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load frozen split indices or generate them with exact stratified alignment."""
        train_path = self.handoff_dir / "train_idx.npy"
        val_path = self.handoff_dir / "val_idx.npy"
        test_path = self.handoff_dir / "test_idx.npy"
        hash_log_path = self.handoff_dir / "splits_hash_log.json"

        # Check nested handoff path if not in root
        if not train_path.exists():
            nested_dir = self.handoff_dir / "safe_main"
            if (nested_dir / "train_idx.npy").exists():
                train_path = nested_dir / "train_idx.npy"
                val_path = nested_dir / "val_idx.npy"
                test_path = nested_dir / "test_idx.npy"
                hash_log_path = nested_dir / "splits_hash_log.json"

        if train_path.exists() and val_path.exists() and test_path.exists():
            logger.info(f"Loading existing split indices from {train_path.parent}...")
            self.train_idx = np.load(train_path)
            self.val_idx = np.load(val_path)
            self.test_idx = np.load(test_path)
        else:
            logger.info("Frozen split arrays not found in handoff dir; generating exact stratified 60/20/20 splits...")
            if self.X_raw is None:
                self.load_raw_dataset()

            n_samples = len(self.y_raw)
            indices = np.arange(n_samples)

            # 60% train, 40% temp
            train_idx, temp_idx = train_test_split(
                indices,
                train_size=0.60,
                random_state=self.random_seed,
                stratify=self.y_raw,
            )

            # 20% val, 20% test (50% of temp)
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=0.50,
                random_state=self.random_seed,
                stratify=self.y_raw[temp_idx],
            )

            self.train_idx = train_idx
            self.val_idx = val_idx
            self.test_idx = test_idx

            # Save in both handoff locations for total interoperability
            target_dirs = [self.handoff_dir]
            if (self.handoff_dir / "safe_main").exists():
                target_dirs.append(self.handoff_dir / "safe_main")

            for tdir in target_dirs:
                tdir.mkdir(parents=True, exist_ok=True)
                np.save(tdir / "train_idx.npy", self.train_idx)
                np.save(tdir / "val_idx.npy", self.val_idx)
                np.save(tdir / "test_idx.npy", self.test_idx)

                # Compute hashes
                h_train = compute_sha256(tdir / "train_idx.npy")
                h_val = compute_sha256(tdir / "val_idx.npy")
                h_test = compute_sha256(tdir / "test_idx.npy")

                hash_dict = {
                    "dataset": "X-IIoTID",
                    "total_samples": int(n_samples),
                    "split_ratios": {"train": 0.60, "val": 0.20, "test": 0.20},
                    "random_seed": self.random_seed,
                    "sha256_hashes": {
                        "train_idx.npy": h_train,
                        "val_idx.npy": h_val,
                        "test_idx.npy": h_test,
                    },
                    "sample_counts": {
                        "train": {
                            "total": len(self.train_idx),
                            "normal": int(np.sum(self.y_raw[self.train_idx] == 0)),
                            "attack": int(np.sum(self.y_raw[self.train_idx] == 1)),
                        },
                        "val": {
                            "total": len(self.val_idx),
                            "normal": int(np.sum(self.y_raw[self.val_idx] == 0)),
                            "attack": int(np.sum(self.y_raw[self.val_idx] == 1)),
                        },
                        "test": {
                            "total": len(self.test_idx),
                            "normal": int(np.sum(self.y_raw[self.test_idx] == 0)),
                            "attack": int(np.sum(self.y_raw[self.test_idx] == 1)),
                        },
                    },
                }

                with open(tdir / "splits_hash_log.json", "w", encoding="utf-8") as f:
                    json.dump(hash_dict, f, indent=2)
                logger.info(f"Saved frozen splits and SHA256 log to {tdir}")

        # Validate counts
        n_tr = len(self.train_idx)
        n_val = len(self.val_idx)
        n_te = len(self.test_idx)
        logger.info(f"Split sizes: Train={n_tr:,}, Val={n_val:,}, Test={n_te:,} (Total: {n_tr + n_val + n_te:,})")
        return self.train_idx, self.val_idx, self.test_idx

    def verify_split_integrity(self) -> Dict[str, Any]:
        """Verify SHA256 hashes against splits_hash_log.json."""
        target_dir = self.handoff_dir
        if not (target_dir / "splits_hash_log.json").exists() and (target_dir / "safe_main" / "splits_hash_log.json").exists():
            target_dir = target_dir / "safe_main"

        hash_log_path = target_dir / "splits_hash_log.json"
        if not hash_log_path.exists():
            raise FileNotFoundError(f"splits_hash_log.json not found in {target_dir}")

        with open(hash_log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        expected = log_data["sha256_hashes"]
        actual_train = compute_sha256(target_dir / "train_idx.npy")
        actual_val = compute_sha256(target_dir / "val_idx.npy")
        actual_test = compute_sha256(target_dir / "test_idx.npy")

        results = {
            "train_idx": {"expected": expected["train_idx.npy"], "actual": actual_train, "match": actual_train == expected["train_idx.npy"]},
            "val_idx": {"expected": expected["val_idx.npy"], "actual": actual_val, "match": actual_val == expected["val_idx.npy"]},
            "test_idx": {"expected": expected["test_idx.npy"], "actual": actual_test, "match": actual_test == expected["test_idx.npy"]},
            "all_passed": bool(
                actual_train == expected["train_idx.npy"]
                and actual_val == expected["val_idx.npy"]
                and actual_test == expected["test_idx.npy"]
            ),
        }
        logger.info(f"Split integrity verification status: {results['all_passed']}")
        return results

    def fit_and_extract_pca_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit PCA on strictly normal training traffic (y=0) and extract 31 features."""
        if self.X_raw is None:
            self.load_raw_dataset()
        if self.train_idx is None:
            self.get_or_create_splits()

        X_train_raw = self.X_raw[self.train_idx]
        y_train_raw = self.y_raw[self.train_idx]
        X_val_raw = self.X_raw[self.val_idx]
        X_test_raw = self.X_raw[self.test_idx]

        # Extract strictly normal training traffic for zero-leakage fitting
        normal_mask = (y_train_raw == 0)
        X_train_normal = X_train_raw[normal_mask]
        logger.info(f"Fitting PCA feature selector on {len(X_train_normal):,} normal training rows...")

        if PCAFeatureSelector is not None:
            self.pca_selector = PCAFeatureSelector(
                variance_threshold=self.pca_variance_threshold,
                top_k=self.k_pca,
                standardize=True,
            )
            self.pca_selector.fit(X_train_normal, feature_names=self.raw_feature_names)
            raw_tr = self.pca_selector.transform(X_train_raw)
            raw_val = self.pca_selector.transform(X_val_raw)
            raw_test = self.pca_selector.transform(X_test_raw)
            self.selected_feature_names = self.pca_selector.selected_feature_names_
        else:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            scaler = StandardScaler()
            X_norm_sc = scaler.fit_transform(X_train_normal)
            pca = PCA(n_components=min(X_train_normal.shape), svd_solver="full")
            pca.fit(X_norm_sc)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            m_idx = min(int(np.searchsorted(cum_var, self.pca_variance_threshold)) + 1, X_train_normal.shape[1])
            loadings_m = pca.components_[:m_idx, :]
            ranking_scores = np.sum(np.abs(loadings_m), axis=0)
            sorted_idx = np.argsort(-ranking_scores)[:self.k_pca]

            raw_tr = X_train_raw[:, sorted_idx]
            raw_val = X_val_raw[:, sorted_idx]
            raw_test = X_test_raw[:, sorted_idx]
            self.selected_feature_names = [self.raw_feature_names[i] for i in sorted_idx]

        # Standardize the 31 features strictly on normal training data (zero-leakage)
        from sklearn.preprocessing import StandardScaler
        feature_scaler = StandardScaler()
        feature_scaler.fit(raw_tr[normal_mask])

        self.X_train_pca = np.clip(feature_scaler.transform(raw_tr), -10.0, 10.0).astype(np.float32)
        self.X_val_pca = np.clip(feature_scaler.transform(raw_val), -10.0, 10.0).astype(np.float32)
        self.X_test_pca = np.clip(feature_scaler.transform(raw_test), -10.0, 10.0).astype(np.float32)

        logger.info(f"Extracted {self.X_train_pca.shape[1]} PCA features.")
        logger.info(f"Top 5 selected features: {self.selected_feature_names[:5]}")

        # Stratify flow volumes
        v_train = self.flow_volumes_raw[self.train_idx]
        v_val = self.flow_volumes_raw[self.val_idx]
        v_test = self.flow_volumes_raw[self.test_idx]

        self.strata_train, self.strata_thresholds = assign_flow_volume_strata(v_train)
        self.strata_val, _ = assign_flow_volume_strata(v_val)
        self.strata_test, _ = assign_flow_volume_strata(v_test)

        return self.X_train_pca, self.X_val_pca, self.X_test_pca

    def get_dataloaders(self, batch_size: int = 512, normal_only_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Return PyTorch DataLoaders for pretraining and evaluation."""
        if self.X_train_pca is None:
            self.fit_and_extract_pca_features()

        y_train = self.y_raw[self.train_idx]
        v_train = self.flow_volumes_raw[self.train_idx]

        if normal_only_train:
            mask = (y_train == 0)
            X_tr = self.X_train_pca[mask]
            y_tr = y_train[mask]
            v_tr = v_train[mask]
            strata_tr = self.strata_train[mask]
        else:
            X_tr = self.X_train_pca
            y_tr = y_train
            v_tr = v_train
            strata_tr = self.strata_train

        train_ds = TabularDataset(X_tr, y_tr, v_tr, strata_tr)
        val_ds = TabularDataset(self.X_val_pca, self.y_raw[self.val_idx], self.flow_volumes_raw[self.val_idx], self.strata_val)
        test_ds = TabularDataset(self.X_test_pca, self.y_raw[self.test_idx], self.flow_volumes_raw[self.test_idx], self.strata_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
