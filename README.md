# BA-MAE: Block-Aware Masked Autoencoder for Industrial IoT Intrusion Detection

**Bachelor Technology Project (BTP)**  
**Indian Institute of Information Technology (IIIT), Pune**  
**Author**: Lead ML Engineer & Cybersecurity Researcher  
**Advisor**: Dr. Shivangi Shukla  
**Benchmark Dataset**: X-IIoTID Intrusion Detection Benchmark (Al-Hawawreh et al., 2022)  
**Baseline Alignment**: SAFE Framework (arXiv:2502.07119)  

---

## Overview

This repository implements **BA-MAE (Block-Aware Masked Autoencoder)**, a self-supervised representation learning framework for tabular Network Intrusion Detection Systems (NIDS). Built on the X-IIoTID benchmark dataset, BA-MAE replaces arbitrary pixel/feature masking with **dependency-aware block masking** and **tail-aware stratum scheduling**, significantly enhancing anomaly detection and probe classification performance under heavy-tailed network traffic.

### Controlled Experimental Ladder (E1 $\to$ E4)

We evaluate BA-MAE across a 5-arm controlled ladder with identical pretraining budgets and 5 fixed random seeds ($42, 43, 44, 45, 46$):

| Arm | Masking Strategy | Masking Budget | Reconstruction Loss | Core Scientific Hypothesis |
|:---|:---|:---|:---|:---|
| **E1** | Random Feature Masking | Fixed 8/31 features | Masked MSE | Baseline self-supervised tabular autoencoding |
| **E2** | Block-Aware Masking | Fixed 8/31 features (whole blocks) | Masked MSE | Isolates benefit of masking correlated feature blocks |
| **E3** | Block-Aware Masking | Fixed 8/31 features (whole blocks) | Masked Huber ($\delta=1.0$) | Robustness against heavy-tailed network traffic outliers |
| **E4** | Block-Aware Tail-Aware | Stratum Schedule (25% to 65%) | Masked Huber ($\delta=1.0$) | Harder reconstruction pretext task for extreme flows |
| **E4-ctrl** | Random Tail-Aware | Stratum Schedule (25% to 65%) | Masked Huber ($\delta=1.0$) | Ablation control isolating block-awareness from schedule |

---

## Repository Structure

```
├── .gitignore                      # Git ignore rules for data, arrays, and checkpoints
├── README.md                       # Project documentation and experimental roadmap
├── EXPERIMENT_LEDGER.md            # Benchmark metrics, ablation results, and statistical significance
├── alpha_0.35_blocks_xiiotid.csv   # Discovered dependency blocks (alpha=0.35)
├── verify_splits.py                # Split integrity verification script
├── dataset.py                      # Data loading, stratification, and feature selection integration
├── blocks.py                       # Dependency matrix calculation, clustering, and dynamic block sampler
├── models.py                       # Tabular MLP MAE architecture, Huber loss, and MLP Probe
├── train.py                        # Controlled ladder training execution (E1 - E4)
├── evaluate.py                     # Evaluation harness (LOF, MLP Probe, bootstrap CI, McNemar)
└── utils.py                        # Statistical testing, seeding, and flow stratification utilities
```

---

## Statistical Protocol & Metric Integrity

- **Sample Partitions**: Exactly 492,500 train, 164,167 validation, and 164,167 test samples (60/20/20 stratified split).
- **Zero Leakage**: All scalers, PCA loadings, and dependency blocks are derived strictly from normal training traffic ($y=0$).
- **Statistical Significance**: Paired 95% bootstrap confidence intervals ($B=1000$) and McNemar's test for parity verification ($E_3$ vs. $E_2$ and $E_4$ vs. $E_4\text{-control}$).
- **Evaluation Heads**:
  - **Path A**: Unsupervised Local Outlier Factor (LOF) calibrated to validation $T_{95}$ percentile.
  - **Path B**: Supervised MLP Probe ($128 \to 64 \to 1$) trained on frozen representations.
