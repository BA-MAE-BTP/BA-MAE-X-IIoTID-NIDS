"""Execute Arm E4 Strategy 2 Upgraded Score Path (Isolation Forest & KNN) across 5 seeds."""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from utils import setup_logger
from dataset import XIIoTIDDataModule
from models import TabularMAE
from evaluate import extract_all_latents, run_path_a_lof, evaluate_metrics

logger = setup_logger("bamae.strategy2")


def run_strategy2_evaluation():
    logger.info("=== STARTING ARM E4 STRATEGY 2 EVALUATION (IForest & KNN) ===")
    
    # 1. Initialize dataset module
    dm = XIIoTIDDataModule()
    dm.fit_and_extract_pca_features()
    
    seeds = [42, 43, 44, 45, 46]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    
    Path("ladder/eval").mkdir(parents=True, exist_ok=True)
    
    results = {
        "LOF": [],
        "IsolationForest": [],
        "KNN": []
    }
    
    score_columns = {}
    y_test_global = None
    
    for seed in seeds:
        ckpt_path = Path(f"ladder/runs/E4_s{seed}/ckpt_last.pt")
        logger.info(f"\n--- Evaluating Arm E4 Seed {seed} ({ckpt_path}) ---")
        
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = TabularMAE(input_dim=31, latent_dim=128, dropout=0.10)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        
        # Extract latents with mask=0
        z_tr_norm, z_val, y_val, z_test, y_test, strata_te = extract_all_latents(model, dm, device=device)
        if y_test_global is None:
            y_test_global = y_test
        
        # Filter benign (normal, y=0) validation latents for Strategy 2 fitting
        normal_val_mask = (y_val == 0)
        z_val_normal = z_val[normal_val_mask]
        logger.info(f"Benign validation latents: {z_val_normal.shape[0]} samples")
        logger.info(f"Test latents: {z_test.shape[0]} samples")
        
        # ---------------------------------------------------------
        # Head 1: LOF Baseline
        # ---------------------------------------------------------
        preds_lof, scores_lof, m_lof = run_path_a_lof(z_tr_norm, z_val, y_val, z_test, y_test)
        results["LOF"].append({
            "seed": seed,
            "f1": m_lof["f1"],
            "roc_auc": m_lof["roc_auc"],
            "pr_auc": m_lof["pr_auc"],
            "precision": m_lof["precision"],
            "recall": m_lof["recall"],
        })
        score_columns[f"score_lof_s{seed}"] = scores_lof
        score_columns[f"pred_lof_s{seed}"] = preds_lof
        
        # ---------------------------------------------------------
        # Head 2: Isolation Forest (200 trees, max_samples=1024)
        # ---------------------------------------------------------
        logger.info("Fitting Isolation Forest (n_estimators=200, max_samples=1024) on benign validation latents...")
        iso_forest = IsolationForest(
            n_estimators=200,
            max_samples=1024,
            random_state=seed,
            n_jobs=-1
        )
        iso_forest.fit(z_val_normal)
        
        # score_samples returns opposite of anomaly score; negative score_samples => higher = more anomalous
        val_normal_if_scores = -iso_forest.score_samples(z_val_normal)
        t95_if = float(np.percentile(val_normal_if_scores, 95.0))
        
        test_if_scores = -iso_forest.score_samples(z_test)
        preds_if = (test_if_scores >= t95_if).astype(int)
        
        m_if = evaluate_metrics(y_test, preds_if, test_if_scores)
        results["IsolationForest"].append({
            "seed": seed,
            "f1": m_if["f1"],
            "roc_auc": m_if["roc_auc"],
            "pr_auc": m_if["pr_auc"],
            "precision": m_if["precision"],
            "recall": m_if["recall"],
            "t95": t95_if
        })
        score_columns[f"score_iforest_s{seed}"] = test_if_scores
        score_columns[f"pred_iforest_s{seed}"] = preds_if
        logger.info(f"Isolation Forest (Seed {seed}) -> F1: {m_if['f1']*100:.2f}%, ROC-AUC: {m_if['roc_auc']*100:.2f}%, PR-AUC: {m_if['pr_auc']*100:.2f}%, T95: {t95_if:.4f}")
        
        # ---------------------------------------------------------
        # Head 3: KNN (k=20, Euclidean metric)
        # ---------------------------------------------------------
        logger.info("Fitting KNN (n_neighbors=20, metric='euclidean') on benign validation latents...")
        knn = NearestNeighbors(
            n_neighbors=20,
            metric="euclidean",
            algorithm="auto",
            n_jobs=-1
        )
        knn.fit(z_val_normal)
        
        # Average distance to 20 nearest benign neighbors as anomaly score
        val_distances, _ = knn.kneighbors(z_val_normal)
        val_normal_knn_scores = np.mean(val_distances, axis=1)
        t95_knn = float(np.percentile(val_normal_knn_scores, 95.0))
        
        # Score test latents in batches to prevent excessive memory usage
        test_knn_scores_list = []
        batch_eval = 10000
        for b_idx in range(0, len(z_test), batch_eval):
            batch_z = z_test[b_idx: b_idx + batch_eval]
            b_dist, _ = knn.kneighbors(batch_z)
            test_knn_scores_list.append(np.mean(b_dist, axis=1))
        test_knn_scores = np.concatenate(test_knn_scores_list)
        preds_knn = (test_knn_scores >= t95_knn).astype(int)
        
        m_knn = evaluate_metrics(y_test, preds_knn, test_knn_scores)
        results["KNN"].append({
            "seed": seed,
            "f1": m_knn["f1"],
            "roc_auc": m_knn["roc_auc"],
            "pr_auc": m_knn["pr_auc"],
            "precision": m_knn["precision"],
            "recall": m_knn["recall"],
            "t95": t95_knn
        })
        score_columns[f"score_knn_s{seed}"] = test_knn_scores
        score_columns[f"pred_knn_s{seed}"] = preds_knn
        logger.info(f"KNN (Seed {seed}) -> F1: {m_knn['f1']*100:.2f}%, ROC-AUC: {m_knn['roc_auc']*100:.2f}%, PR-AUC: {m_knn['pr_auc']*100:.2f}%, T95: {t95_knn:.4f}")

    # 3. Save raw scores to parquet
    logger.info("\nSaving raw scores to ladder/eval/E4_strategy2_scores.parquet...")
    df_scores = pd.DataFrame(score_columns)
    df_scores["y_true"] = y_test_global
    df_scores["stratum"] = strata_te
    parquet_path = Path("ladder/eval/E4_strategy2_scores.parquet")
    df_scores.to_parquet(parquet_path, engine="pyarrow", index=False)
    logger.info(f"Successfully saved {len(df_scores):,} rows to {parquet_path}")

    # 4. Generate Summary Table (Mean +/- Std across 5 seeds)
    summary_table = []
    for model_name, runs in results.items():
        f1_vals = [r["f1"] for r in runs]
        roc_vals = [r["roc_auc"] for r in runs]
        pr_vals = [r["pr_auc"] for r in runs]
        prec_vals = [r["precision"] for r in runs]
        rec_vals = [r["recall"] for r in runs]
        
        summary_table.append({
            "Anomaly Head": model_name,
            "F1-Score": f"{np.mean(f1_vals)*100:.2f} +/- {np.std(f1_vals)*100:.2f}%",
            "ROC-AUC": f"{np.mean(roc_vals)*100:.2f} +/- {np.std(roc_vals)*100:.2f}%",
            "PR-AUC": f"{np.mean(pr_vals)*100:.2f} +/- {np.std(pr_vals)*100:.2f}%",
            "Precision": f"{np.mean(prec_vals)*100:.2f} +/- {np.std(prec_vals)*100:.2f}%",
            "Recall": f"{np.mean(rec_vals)*100:.2f} +/- {np.std(rec_vals)*100:.2f}%",
        })
        
    df_summary = pd.DataFrame(summary_table)
    print("\n" + "="*85)
    print("ARM E4 BENCHMARK: STRATEGY 2 SCORE PATH COMPARISON (Mean +/- Std Across 5 Seeds)")
    print("="*85)
    print(df_summary.to_string(index=False))
    print("="*85 + "\n")
    
    # Save markdown summary
    df_summary.to_markdown("ladder/eval/E4_strategy2_summary.md", index=False)
    logger.info("Saved summary table to ladder/eval/E4_strategy2_summary.md")


if __name__ == "__main__":
    run_strategy2_evaluation()
