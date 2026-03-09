from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sdr2hdr.dataset import HDRSDRPairDataset
from sdr2hdr.model import EnhancementUNet


def total_variation_loss(tensor: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).mean()
    dy = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).mean()
    return dx + dy


def compute_loss(pred: torch.Tensor, target_maps: torch.Tensor, clip_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    confidence = 1.0 - clip_mask * 0.8
    pred_exp, pred_con, pred_pro = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    tgt_exp, tgt_con, tgt_pro = target_maps[:, 0:1], target_maps[:, 1:2], target_maps[:, 2:3]
    protected_weight = 1.0 + torch.clamp(tgt_pro, min=0.0) * 1.5
    overdrive = torch.clamp(pred_exp - tgt_exp, min=0.0)
    l_exp = (confidence * protected_weight * torch.abs(pred_exp - tgt_exp)).mean() + (protected_weight * overdrive).mean() * 0.5
    l_con = (confidence * (1.0 + torch.clamp(tgt_pro, min=0.0)) * torch.abs(pred_con - tgt_con)).mean()
    l_pro = torch.abs(pred_pro - tgt_pro).mean()
    l_tv = total_variation_loss(pred_exp) * 0.01
    total = 1.0 * l_exp + 0.5 * l_con + 0.75 * l_pro + l_tv
    return total, {"exp": l_exp, "con": l_con, "pro": l_pro, "tv": l_tv}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train enhancement map estimator.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    train_source = HDRSDRPairDataset(args.data_dir, patch_size=args.patch_size, training=True)
    val_source = HDRSDRPairDataset(args.data_dir, patch_size=args.patch_size, training=False)
    indices = list(range(len(train_source)))
    val_size = max(1, len(indices) // 10)
    train_indices = indices[:-val_size] or indices
    val_indices = indices[-val_size:]
    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(val_source, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = EnhancementUNet().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    warmup = LinearLR(optimizer, start_factor=0.2, total_iters=3)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 3))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])
    scaler = torch.amp.GradScaler(device=device.type, enabled=device.type != "cpu")
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train {epoch+1}/{args.epochs}"):
            sdr_linear = batch["sdr_linear"].to(device)
            target_maps = batch["target_maps"].to(device)
            clip_mask = batch["clip_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type != "cpu"):
                pred = model(sdr_linear)
                loss, _ = compute_loss(pred, target_maps, clip_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu())
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val {epoch+1}/{args.epochs}"):
                sdr_linear = batch["sdr_linear"].to(device)
                target_maps = batch["target_maps"].to(device)
                clip_mask = batch["clip_mask"].to(device)
                pred = model(sdr_linear)
                loss, losses = compute_loss(pred, target_maps, clip_mask)
                val_loss += float(loss.detach().cpu())
        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, output_dir / "best.pt")

    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
