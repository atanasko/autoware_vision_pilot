"""
Main training loop for AutoDrive (ZOD dataset).

All outputs are written under {zod_root}/training/autodrive/:
    checkpoints/AutoDrive_last.pth   — saved every epoch (full state, restorable)
    checkpoints/AutoDrive_best.pth   — saved when val loss improves
    tensorboard/                     — TensorBoard event files

Usage — fresh training:
    python Models/training/train_auto_drive.py \\
        --root ~/Downloads/data/zod

Usage — resume after crash / keyboard interrupt:
    python Models/training/train_auto_drive.py \\
        --root ~/Downloads/data/zod \\
        --resume  (auto-loads AutoDrive_last.pth from the training dir)

    or point at any checkpoint explicitly:
        --resume --checkpoint ~/Downloads/data/zod/training/autodrive/checkpoints/AutoDrive_epoch5.pth

Optional:
    [--epochs 50] [--batch-size 16] [--workers 2]

TensorBoard:
    tensorboard --logdir ~/Downloads/data/zod/training/autodrive/tensorboard/

    Scalars
    -------
    Loss/train_total          — raw per-step loss       (noisy, every 200 steps)
    Loss/train_avg_total      — epoch-averaged loss      (smooth, once per epoch)
    Loss/train_avg_distance   — epoch-averaged dist L1
    Loss/train_avg_curvature  — epoch-averaged curv L1
    Loss/train_avg_flag       — epoch-averaged flag BCE
    Loss/val_*                — validation breakdown
    Metrics/flag_acc_%        — CIPO flag accuracy 0–100
    Metrics/dist_mae_m        — distance MAE in real metres
    Metrics/lr                — learning rate
    Visualization/sample      — annotated frame every 1000 steps
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Models.data_utils.load_data_auto_drive import LoadDataAutoDrive
from Models.training.auto_drive_trainer import AutoDriveTrainer


def _collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _run_val(trainer: AutoDriveTrainer, loader: DataLoader):
    """Returns averaged (total, dist, curv, flag, flag_acc_pct, dist_mae_m)."""
    total = dist = curv = flag = acc = mae = 0.0
    n = 0
    for batch in loader:
        t, d, c, f, a, m = trainer.validate(batch)
        total += t; dist += d; curv += c; flag += f; acc += a; mae += m
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return total/n, dist/n, curv/n, flag/n, acc/n, mae/n


def main():
    parser = ArgumentParser()
    parser.add_argument("--root",       required=True,
                        help="ZOD dataset root — all outputs go under {root}/training/autodrive/")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from AutoDrive_last.pth in the training dir")
    parser.add_argument("--checkpoint", default="",
                        help="Explicit checkpoint path to resume from (implies --resume)")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers",    type=int, default=2)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Output directories — everything under zod_root/training/autodrive/
    # ------------------------------------------------------------------
    train_root  = Path(args.root) / "training" / "autodrive"
    ckpt_dir    = train_root / "checkpoints"
    tb_dir      = train_root / "tensorboard"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    ckpt_last = str(ckpt_dir / "AutoDrive_last.pth")
    ckpt_best = str(ckpt_dir / "AutoDrive_best.pth")

    print(f"Checkpoints : {ckpt_dir}")
    print(f"TensorBoard : {tb_dir}")
    print(f"  tensorboard --logdir {tb_dir}")

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    data = LoadDataAutoDrive(args.root)

    train_loader = DataLoader(
        data.train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=_collate, drop_last=True,
    )
    val_loader = DataLoader(
        data.val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=_collate,
    )
    test_loader = DataLoader(
        data.test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=_collate,
    )

    steps_per_epoch = len(train_loader)
    print(f"Train: {len(data.train):,}  Val: {len(data.val):,}  Test: {len(data.test):,} pairs")
    print(f"Steps per epoch: {steps_per_epoch:,}")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = AutoDriveTrainer(tensorboard_dir=str(tb_dir))
    trainer.zero_grad()

    # Resume state
    start_epoch   = 0
    global_step   = 0
    best_val_loss = float('inf')

    # Determine checkpoint to load
    resume_path = ""
    if args.checkpoint:
        if Path(args.checkpoint).exists():
            resume_path = args.checkpoint
        else:
            print(f"  WARNING: --checkpoint path not found: {args.checkpoint}")
            print("  Starting fresh (no checkpoint loaded).")
    elif args.resume:
        if Path(ckpt_last).exists():
            resume_path = ckpt_last
        else:
            print("  --resume: no AutoDrive_last.pth found yet, starting fresh.")

    if resume_path:
        start_epoch, global_step, best_val_loss = trainer.load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}  (global step starts at {global_step})")

        # LR schedule
        if epoch < 5:
            trainer.set_learning_rate(1e-3)
        elif epoch < 20:
            trainer.set_learning_rate(1e-4)
        else:
            trainer.set_learning_rate(1e-5)

        trainer.set_train_mode()
        trainer.reset_averages()

        p_bar = tqdm.tqdm(train_loader, total=steps_per_epoch,
                          desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in p_bar:
            trainer.set_batch(batch)
            trainer.run_model()
            trainer.loss_backward()
            trainer.run_optimizer()

            global_step += 1

            p_bar.set_description(
                f"Epoch {epoch+1}/{args.epochs}  "
                f"loss {trainer.avg_total.avg:.4f}  "
                f"d {trainer.avg_distance.avg:.4f}  "
                f"c {trainer.avg_curvature.avg:.4f}  "
                f"f {trainer.avg_flag.avg:.4f}"
            )

            if global_step % 200 == 0:
                trainer.log_train_step(global_step)

            if global_step % 1000 == 0:
                trainer.save_visualization(global_step)

        trainer.log_train_epoch(epoch + 1)

        # ------------------------------------------------------------------
        # Save last checkpoint (full state — always safe to resume from)
        # ------------------------------------------------------------------
        trainer.save_checkpoint(ckpt_last, epoch + 1, global_step, best_val_loss)

        # Also save a named copy so you can roll back to a specific epoch
        ckpt_epoch = str(ckpt_dir / f"AutoDrive_epoch{epoch+1:03d}.pth")
        trainer.save_checkpoint(ckpt_epoch, epoch + 1, global_step, best_val_loss)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        print("  Validating...")
        trainer.set_eval_mode()
        with torch.no_grad():
            v_total, v_dist, v_curv, v_flag, v_acc, v_mae = _run_val(trainer, val_loader)

        trainer.log_val_epoch(v_total, v_dist, v_curv, v_flag, v_acc, v_mae, epoch + 1)

        print(
            f"  [Val] loss {v_total:.4f}  "
            f"d {v_dist:.4f}  c {v_curv:.4f}  f {v_flag:.4f}  "
            f"flag_acc {v_acc:.1f}%  dist_mae {v_mae:.1f} m"
        )

        if v_total < best_val_loss:
            best_val_loss = v_total
            trainer.save_checkpoint(ckpt_best, epoch + 1, global_step, best_val_loss)
            print(f"  *** New best val loss: {best_val_loss:.4f} → AutoDrive_best.pth")

        trainer.set_train_mode()

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print("\nTest set evaluation...")
    trainer.set_eval_mode()
    with torch.no_grad():
        t_total, t_dist, t_curv, t_flag, t_acc, t_mae = _run_val(trainer, test_loader)
    trainer.log_test(t_total, t_dist, t_curv, t_flag, t_acc, t_mae)
    print(
        f"[Test] loss {t_total:.4f}  "
        f"d {t_dist:.4f}  c {t_curv:.4f}  f {t_flag:.4f}  "
        f"flag_acc {t_acc:.1f}%  dist_mae {t_mae:.1f} m"
    )

    trainer.cleanup()


if __name__ == "__main__":
    main()
