"""Controlled ladder training execution script (E1 -> E2 -> E3 -> E4 -> E4-control) across 5 seeds."""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import setup_logger, set_seed
from dataset import XIIoTIDDataModule, TabularDataset
from blocks import ExactCountDynamicBlockSampler, sample_random_mask
from models import TabularMAE, MaskedMSELoss, MaskedHuberLoss

logger = setup_logger("bamae.train")

# Stratum-dependent masking schedule: Body ~25%, Light ~35%, Moderate ~45%, Heavy ~55%, Extreme ~65%
STRATUM_SCHEDULE_31: Dict[int, int] = {
    0: 8,   # Body (~25% of 31)
    1: 11,  # Light (~35% of 31)
    2: 14,  # Moderate (~45% of 31)
    3: 17,  # Heavy (~55% of 31)
    4: 20,  # Extreme (~65% of 31)
}


def create_batch_mask(
    batch_features: torch.Tensor,
    batch_strata: torch.Tensor,
    masking_type: str,
    target_budget: int,
    block_sampler: Optional[ExactCountDynamicBlockSampler],
    rng: np.random.RandomState,
    device: torch.device,
) -> torch.Tensor:
    """Generate exact binary mask for batch according to experimental arm specification."""
    batch_size, n_features = batch_features.shape

    if masking_type == "random":
        np_mask = sample_random_mask(batch_size=batch_size, n_features=n_features, target_budget=target_budget, rng=rng)
    elif masking_type == "block":
        if block_sampler is None:
            raise ValueError("block_sampler required for block masking")
        np_mask = block_sampler.sample_mask(batch_size=batch_size, target_budget=target_budget, rng=rng)
    elif masking_type == "block_stratum":
        if block_sampler is None:
            raise ValueError("block_sampler required for block_stratum masking")
        np_mask = np.zeros((batch_size, n_features), dtype=np.float32)
        strata_arr = batch_strata.cpu().numpy()
        for s_idx, budget in STRATUM_SCHEDULE_31.items():
            row_idx = np.where(strata_arr == s_idx)[0]
            if len(row_idx) > 0:
                s_masks = block_sampler.sample_mask(batch_size=len(row_idx), target_budget=budget, rng=rng)
                np_mask[row_idx] = s_masks
    elif masking_type == "random_stratum":
        np_mask = np.zeros((batch_size, n_features), dtype=np.float32)
        strata_arr = batch_strata.cpu().numpy()
        for s_idx, budget in STRATUM_SCHEDULE_31.items():
            row_idx = np.where(strata_arr == s_idx)[0]
            if len(row_idx) > 0:
                s_masks = sample_random_mask(batch_size=len(row_idx), n_features=n_features, target_budget=budget, rng=rng)
                np_mask[row_idx] = s_masks
    else:
        raise ValueError(f"Unknown masking_type: {masking_type}")

    return torch.as_tensor(np_mask, dtype=torch.float32, device=device)


def train_single_arm(
    arm_name: str,
    seed: int,
    data_module: XIIoTIDDataModule,
    block_sampler: ExactCountDynamicBlockSampler,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    output_dir: Path | str = "checkpoints",
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Execute single training run for specified arm and random seed under matched pretraining budget."""
    set_seed(seed)
    rng = np.random.RandomState(seed)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info(f"Training Arm '{arm_name}' | Seed: {seed} | Device: {device} | Epochs: {epochs}")

    # Parse arm specifications
    if arm_name == "E1":
        masking_type = "random"
        loss_type = "mse"
        target_budget = 8
    elif arm_name == "E2":
        masking_type = "block"
        loss_type = "mse"
        target_budget = 8
    elif arm_name == "E3":
        masking_type = "block"
        loss_type = "huber"
        target_budget = 8
    elif arm_name == "E4":
        masking_type = "block_stratum"
        loss_type = "huber"
        target_budget = 8
    elif arm_name == "E4_control":
        masking_type = "random_stratum"
        loss_type = "huber"
        target_budget = 8
    else:
        raise ValueError(f"Unknown arm name: {arm_name}")

    # Loss function
    if loss_type == "mse":
        criterion = MaskedMSELoss().to(device)
    else:
        criterion = MaskedHuberLoss(delta=1.0).to(device)

    # Initialize model
    model = TabularMAE(input_dim=31, latent_dim=128, dropout=0.10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Get data loader (strictly normal training data)
    train_loader, val_loader, _ = data_module.get_dataloaders(batch_size=batch_size, normal_only_train=True)

    epoch_losses = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_x, _, _, batch_strata in train_loader:
            batch_x = batch_x.to(device)
            batch_strata = batch_strata.to(device)

            mask = create_batch_mask(
                batch_features=batch_x,
                batch_strata=batch_strata,
                masking_type=masking_type,
                target_budget=target_budget,
                block_sampler=block_sampler,
                rng=rng,
                device=device,
            )

            recon, _ = model(batch_x, mask=mask)
            loss = criterion(recon, batch_x, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)

        if epoch % 2 == 0 or epoch == epochs:
            logger.info(f"  [{arm_name} s{seed}] Epoch {epoch:2d}/{epochs:2d} - Train Loss: {avg_loss:.5f}")

    elapsed = time.time() - start_time
    logger.info(f"Finished {arm_name} seed {seed} in {elapsed:.2f}s (Final Loss: {epoch_losses[-1]:.5f})")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_p / f"{arm_name}_seed_{seed}.pt"
    torch.save({
        "arm": arm_name,
        "seed": seed,
        "epochs": epochs,
        "loss_type": loss_type,
        "masking_type": masking_type,
        "state_dict": model.state_dict(),
        "final_loss": epoch_losses[-1],
        "losses": epoch_losses,
        "elapsed_seconds": elapsed,
    }, checkpoint_path)

    return {
        "arm": arm_name,
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "final_loss": epoch_losses[-1],
        "elapsed_seconds": elapsed,
        "model": model,
    }


def execute_ladder(
    arms: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    epochs: int = 10,
    batch_size: int = 512,
    output_dir: str = "checkpoints",
) -> Dict[str, Any]:
    """Execute complete controlled ladder across all arms and seeds."""
    if arms is None:
        arms = ["E1", "E2", "E3", "E4", "E4_control"]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    logger.info("=" * 70)
    logger.info(f"EXECUTING CONTROLLED PRETRAINING LADDER: {arms}")
    logger.info(f"Seeds: {seeds} | Epochs per run: {epochs} | Batch size: {batch_size}")
    logger.info("=" * 70)

    # Load data module and blocks
    dm = XIIoTIDDataModule()
    dm.fit_and_extract_pca_features()

    blocks_csv = Path("alpha_0.35_blocks_xiiotid.csv")
    if not blocks_csv.exists():
        raise FileNotFoundError("alpha_0.35_blocks_xiiotid.csv must be generated first")

    df_blocks = pd.read_csv(blocks_csv)
    blocks: Dict[int, List[int]] = {}
    for _, row in df_blocks.iterrows():
        blocks.setdefault(int(row["block_id"]), []).append(int(row["feature_idx"]))

    sampler = ExactCountDynamicBlockSampler(blocks, n_features=31)

    ladder_results = {}
    for arm in arms:
        ladder_results[arm] = []
        for seed in seeds:
            res = train_single_arm(
                arm_name=arm,
                seed=seed,
                data_module=dm,
                block_sampler=sampler,
                epochs=epochs,
                batch_size=batch_size,
                output_dir=output_dir,
            )
            ladder_results[arm].append(res)

    logger.info("All controlled ladder pretraining runs completed successfully.")
    return ladder_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BA-MAE Controlled Pretraining Ladder")
    parser.add_argument("--arms", nargs="+", default=["E1", "E2", "E3", "E4", "E4_control"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    execute_ladder(
        arms=args.arms,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
