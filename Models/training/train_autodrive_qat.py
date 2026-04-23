"""
Quantization-Aware Training (QAT) for AutoDrive — PyTorch 2 Export / XNNPACK flow.

After every epoch this script produces:
  1. QAT prepared checkpoint  — fake-quantize model, still float weights, resumable
  2. Converted INT8 checkpoint — convert_pt2e output, ready for INT8 ONNX export

Both are validated every epoch so TensorBoard shows the quantization gap live.
The best INT8 model (by val loss) is kept separately.

Dependencies (install once):
  pip install torchao executorch

──────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────
  python Models/training/train_autodrive_qat.py \\
      --root ~/data/zod \\
      --checkpoint ~/data/zod/training/autodrive/run002/checkpoints/AutoDrive_best.pth \\
      --epochs 10 \\
      --batch-size 8 \\
      --workers 2 \\
      --export-onnx

──────────────────────────────────────────────────────────────────
Outputs  →  {root}/training/autodrive_qat/{run_name}/
──────────────────────────────────────────────────────────────────
  checkpoints/
    AutoDrive_qat_last.pth              QAT prepared   (latest epoch, resumable)
    AutoDrive_qat_best.pth              QAT prepared   (best val — prepared model)
    AutoDrive_qat_converted_best.pth    INT8 converted (best val — int8 model)
    AutoDrive_qat_converted_final.pth   INT8 converted (last epoch, use for ONNX)
    epochs/
      AutoDrive_qat_ep{N:03d}.pth           QAT prepared  checkpoint per epoch
      AutoDrive_qat_int8_ep{N:03d}.pth      INT8 converted checkpoint per epoch
  tensorboard/

TensorBoard:
  tensorboard --logdir {root}/training/autodrive_qat/{run_name}/tensorboard/

  Tags:
    Loss/train_*              per-step train losses
    Loss/train_avg_*          epoch-averaged train losses
    Loss/val_prepared_*       validation on QAT-prepared model (fake-quantize, float)
    Loss/val_int8_*           validation on INT8-converted model
    Metrics/prepared_*        steer_mae, dist_mae, flag_acc for prepared model
    Metrics/int8_*            same metrics for INT8 model
    Metrics/quant_gap_*       prepared − int8  (shows quantization degradation)
    Metrics/lr
    Info/model_size_float_MB  float model size in MB
    Info/model_size_int8_MB   INT8 converted model size in MB
"""

import copy
import math
import os
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── XNNPACKQuantizer (XNNPACK backend) ────────────────────────────────────
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

# ── torchao PT2E core APIs ─────────────────────────────────────────────────
import torchao.quantization.pt2e as pt2e_utils
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e

# ── Repo imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Models.model_components.autodrive.autodrive_network import AutoDrive
from Models.data_utils.load_data_auto_drive import LoadDataAutoDrive, CURV_SCALE
from Models.training.auto_drive_trainer import AverageMeter


# ─────────────────────────────────────────────────────────────────────────────
# Constants — kept identical to AutoDriveTrainer
# ─────────────────────────────────────────────────────────────────────────────
_CIPO_POS_WEIGHT       = torch.tensor([0.295 / 0.705])
_CURV_LOSS_W           = 2.0
_DIST_LOSS_W           = 2.0
_FLAG_VAL_THRESHOLD    = 0.65
_WHEELBASE_M           = 2.984
_STEERING_COLUMN_RATIO = 16.8


def _collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_loss(
    model_out: tuple,
    d_norm_gt: torch.Tensor,
    curvature_gt: torch.Tensor,
    flag_gt: torch.Tensor,
    dist_mask: torch.Tensor,
    l1: nn.L1Loss,
    bce: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, float]:
    """Total loss + component scalars. Mirrors AutoDriveTrainer joint-mode loss."""
    d_pred, curv_pred, flag_logits = model_out

    loss_c = l1(curv_pred, curvature_gt)

    if dist_mask.any():
        mask_idx = dist_mask.unsqueeze(1)
        loss_d   = l1(d_pred[mask_idx], d_norm_gt[mask_idx])
    else:
        loss_d = torch.tensor(0.0, device=device)

    loss_f = bce(flag_logits, flag_gt)
    total  = _CURV_LOSS_W * loss_c + _DIST_LOSS_W * loss_d + loss_f
    return total, loss_c.item(), loss_d.item(), loss_f.item()


@torch.no_grad()
def _compute_val_metrics(
    model_out: tuple,
    d_norm_gt: torch.Tensor,
    curvature_gt: torch.Tensor,
    flag_gt: torch.Tensor,
    dist_mask: torch.Tensor,
    l1: nn.L1Loss,
    bce: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> dict:
    """Returns a dict with all validation metrics. Mirrors AutoDriveTrainer.validate()."""
    d_pred, curv_pred, flag_logits = model_out

    loss_c = l1(curv_pred, curvature_gt)
    if dist_mask.any():
        mask_idx   = dist_mask.unsqueeze(1)
        loss_d     = l1(d_pred[mask_idx], d_norm_gt[mask_idx])
        dist_mae_m = (150.0 * (d_pred[mask_idx] - d_norm_gt[mask_idx]).abs()).mean().item()
    else:
        loss_d     = torch.tensor(0.0, device=device)
        dist_mae_m = 0.0

    loss_f = bce(flag_logits, flag_gt)
    total  = loss_c + loss_d + loss_f

    flag_prob  = torch.sigmoid(flag_logits)
    pred_label = (flag_prob > _FLAG_VAL_THRESHOLD).float()
    flag_acc   = (pred_label == flag_gt).float().mean().item() * 100.0

    curv_pred_m    = curv_pred * CURV_SCALE
    curv_gt_m      = curvature_gt * CURV_SCALE
    steer_scale    = _STEERING_COLUMN_RATIO * (180.0 / math.pi)
    steer_pred_deg = torch.atan(curv_pred_m * _WHEELBASE_M) * steer_scale
    steer_gt_deg   = torch.atan(curv_gt_m   * _WHEELBASE_M) * steer_scale
    steer_mae_deg  = (steer_pred_deg - steer_gt_deg).abs().mean().item()

    return dict(
        total=total.item(), dist=loss_d.item(), curv=loss_c.item(), flag=loss_f.item(),
        flag_acc=flag_acc, dist_mae_m=dist_mae_m, steer_mae_deg=steer_mae_deg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full validation pass over a loader
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_val(
    model: torch.nn.Module,
    loader: DataLoader,
    l1: nn.L1Loss,
    bce: nn.BCEWithLogitsLoss,
    device: torch.device,
    label: str = "",
) -> dict:
    """
    Average validation metrics across the entire loader.
    The caller is responsible for setting the model to eval mode before calling.
    """
    keys = ("total", "dist", "curv", "flag", "flag_acc", "dist_mae_m", "steer_mae_deg")
    sums = {k: 0.0 for k in keys}
    n    = 0

    p_bar = tqdm.tqdm(loader, desc=f"  Val [{label}]", leave=False)
    for batch in p_bar:
        img_prev  = batch["img_prev"].to(device)
        img_curr  = batch["img_curr"].to(device)
        d_norm_gt = batch["d_norm"].unsqueeze(1).to(device)
        curv_gt   = batch["curvature"].unsqueeze(1).to(device)
        flag_gt   = batch["flag"].unsqueeze(1).to(device)
        dist_mask = batch["dist_mask"].to(device)

        out = model(img_prev, img_curr)
        m   = _compute_val_metrics(out, d_norm_gt, curv_gt, flag_gt,
                                   dist_mask, l1, bce, device)
        for k in keys:
            sums[k] += m[k]
        n += 1

    if n == 0:
        return {k: 0.0 for k in keys}
    return {k: sums[k] / n for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Model size utility
# ─────────────────────────────────────────────────────────────────────────────

def _model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in MB by saving to a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp = f.name
    try:
        torch.save(model.state_dict(), tmp)
        size_mb = os.path.getsize(tmp) / 1e6
    finally:
        os.remove(tmp)
    return size_mb


# ─────────────────────────────────────────────────────────────────────────────
# PT2E export + XNNPACKQuantizer prepare
# ─────────────────────────────────────────────────────────────────────────────

def _export_and_prepare_xnnpack(
    float_model: AutoDrive,
    device: torch.device,
) -> torch.nn.Module:
    """
    Step 1 — Export float model to an ATen graph (torch.export).
    Step 2 — Prepare for QAT using XNNPACKQuantizer (symmetric INT8).
    Returns the QAT-prepared GraphModule with fake-quantize nodes inserted.
    """
    float_model.eval()
    float_model.to(device)

    # Batch=2 so the exporter sees a truly dynamic batch axis (batch=1 gets
    # specialized as a compile-time constant by the exporter).
    img_prev_ex = torch.randn(2, 3, 512, 1024, device=device)
    img_curr_ex = torch.randn(2, 3, 512, 1024, device=device)
    example_inputs = (img_prev_ex, img_curr_ex)

    batch_dim = torch.export.Dim("batch", min=2, max=128)
    dynamic_shapes = (
        {0: batch_dim},   # img_prev  (B, 3, 512, 1024)
        {0: batch_dim},   # img_curr  (B, 3, 512, 1024)
    )

    print("  [1/2] Exporting AutoDrive to ATen graph …")
    exported = torch.export.export(
        float_model,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    ).module()

    # XNNPACKQuantizer — XNNPACK symmetric INT8
    #   is_qat=True   → fuses Conv2d+BN before inserting fake-quantizes
    #   is_per_channel → per-channel weight quantization (better accuracy)
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_qat=True)
    )

    print("  [2/2] Inserting fake-quantize nodes (XNNPACK symmetric INT8) …")
    prepared = prepare_qat_pt2e(exported, quantizer)
    return prepared


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def _get_qat_lr(epoch: int) -> float:
    """
    Epoch 0–4  : 1e-5  — observers calibrate with gentle updates
    Epoch 5+   : 5e-6  — fine-tune with frozen observer stats
    """
    return 1e-5 if epoch < 5 else 5e-6


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tb_val(writer: SummaryWriter, tag_prefix: str, m: dict, step: int):
    """Log a full validation metrics dict under tag_prefix."""
    writer.add_scalar(f"Loss/{tag_prefix}_total",       m["total"],       step)
    writer.add_scalar(f"Loss/{tag_prefix}_curvature",   m["curv"],        step)
    writer.add_scalar(f"Loss/{tag_prefix}_distance",    m["dist"],        step)
    writer.add_scalar(f"Loss/{tag_prefix}_flag",        m["flag"],        step)
    writer.add_scalar(f"Metrics/{tag_prefix}_flag_acc", m["flag_acc"],    step)
    writer.add_scalar(f"Metrics/{tag_prefix}_dist_mae", m["dist_mae_m"],  step)
    writer.add_scalar(f"Metrics/{tag_prefix}_steer_mae",m["steer_mae_deg"],step)


def _tb_quant_gap(writer: SummaryWriter, prepared: dict, int8: dict, step: int):
    """Log prepared − int8 metric deltas so the quantization gap is visible."""
    writer.add_scalar("Metrics/quant_gap_total",     prepared["total"]       - int8["total"],       step)
    writer.add_scalar("Metrics/quant_gap_steer_mae", prepared["steer_mae_deg"] - int8["steer_mae_deg"], step)
    writer.add_scalar("Metrics/quant_gap_dist_mae",  prepared["dist_mae_m"]  - int8["dist_mae_m"],  step)
    writer.add_scalar("Metrics/quant_gap_flag_acc",  prepared["flag_acc"]    - int8["flag_acc"],    step)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(
        description="Quantization-Aware Training (QAT) for AutoDrive — XNNPACK INT8"
    )
    parser.add_argument("--root",       required=True,
                        help="ZOD dataset root")
    parser.add_argument("--checkpoint", required=True,
                        help="Float AutoDrive .pth checkpoint to start QAT from")
    parser.add_argument("--run-name",   default="",
                        help="Sub-folder name (default: qat001, qat002, …)")
    parser.add_argument("--epochs",     type=int, default=10,
                        help="QAT epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers",    type=int, default=2)
    parser.add_argument("--qat-lr",     type=float, default=0.0,
                        help="Fixed QAT learning rate (0 = use built-in schedule)")
    parser.add_argument("--observer-freeze-epoch", type=int, default=4,
                        help="Freeze observer range stats after epoch N (default: 4)")
    parser.add_argument("--bn-freeze-epoch",       type=int, default=3,
                        help="Freeze BatchNorm running stats after epoch N (default: 3)")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export final INT8 model to ONNX after training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*64}")
    print(f"  AutoDrive QAT  —  XNNPACK symmetric INT8")
    print(f"  Device        : {device}")
    print(f"  Float ckpt    : {args.checkpoint}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"{'='*64}\n")

    # ── Output directories ────────────────────────────────────────────────
    base_dir = Path(args.root) / "training" / "autodrive_qat"
    if args.run_name:
        run_name = args.run_name
    else:
        existing = sorted(base_dir.glob("qat[0-9][0-9][0-9]"))
        run_name = f"qat{len(existing) + 1:03d}"

    run_dir    = base_dir / run_name
    ckpt_dir   = run_dir / "checkpoints"
    epoch_dir  = ckpt_dir / "epochs"
    tb_dir     = run_dir / "tensorboard"
    for d in (ckpt_dir, epoch_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    ckpt_last          = ckpt_dir / "AutoDrive_qat_last.pth"
    ckpt_best_prepared = ckpt_dir / "AutoDrive_qat_best.pth"
    ckpt_best_int8     = ckpt_dir / "AutoDrive_qat_converted_best.pth"
    ckpt_final_int8    = ckpt_dir / "AutoDrive_qat_converted_final.pth"

    print(f"Run         : {run_dir}")
    print(f"TensorBoard : tensorboard --logdir {tb_dir}\n")

    # ── Data ────────────────────────────────────────────────────────────
    data = LoadDataAutoDrive(args.root)
    train_loader = DataLoader(
        data.train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=_collate, drop_last=True,
    )
    val_loader = DataLoader(
        data.val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=_collate,
    )
    print(f"Dataset  →  Train: {len(data.train):,}   Val: {len(data.val):,}")

    # ── Load float model + measure its size ─────────────────────────────
    print(f"\nLoading float checkpoint ← {args.checkpoint}")
    float_model = AutoDrive()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        float_model.load_state_dict(ckpt["model"])
    else:
        float_model.load_state_dict(ckpt)
    float_model.to(device)
    float_size_mb = _model_size_mb(float_model)
    print(f"  Float model size : {float_size_mb:.1f} MB")

    # ── Export + prepare QAT with XNNPACKQuantizer ───────────────────────
    print()
    prepared_model = _export_and_prepare_xnnpack(float_model, device)
    prepared_model.to(device)
    print(f"  QAT model ready  : {type(prepared_model).__name__}")

    # ── Optimizer + loss functions ───────────────────────────────────────
    lr_init   = args.qat_lr if args.qat_lr > 0 else _get_qat_lr(0)
    optimizer = torch.optim.Adam(
        prepared_model.parameters(), lr=lr_init, weight_decay=1e-5
    )
    l1  = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss(pos_weight=_CIPO_POS_WEIGHT.to(device))

    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_scalar("Info/model_size_float_MB", float_size_mb, 0)

    best_prepared_val = float("inf")
    best_int8_val     = float("inf")
    global_step       = 0

    # ────────────────────────────────────────────────────────────────────
    # QAT training loop
    # ────────────────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        print(f"\n{'='*64}")
        print(f"  QAT  Epoch {epoch+1:>3d}/{args.epochs}   "
              f"[observer_freeze≥{args.observer_freeze_epoch}, "
              f"bn_freeze≥{args.bn_freeze_epoch}]")
        print(f"{'='*64}")

        new_lr = args.qat_lr if args.qat_lr > 0 else _get_qat_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        writer.add_scalar("Metrics/lr", new_lr, epoch + 1)

        # ── Optionally freeze observer / BN stats ─────────────────────
        if epoch >= args.observer_freeze_epoch:
            print(f"  ▸ Freezing observer range stats (epoch ≥ {args.observer_freeze_epoch})")
            prepared_model.apply(pt2e_utils.disable_observer)

        if epoch >= args.bn_freeze_epoch:
            print(f"  ▸ Freezing BatchNorm running stats (epoch ≥ {args.bn_freeze_epoch})")
            _bn_targets = {
                torch.ops.aten._native_batch_norm_legit.default,
                torch.ops.aten.cudnn_batch_norm.default,
                torch.ops.aten._native_batch_norm_legit_no_training.default,
            }
            for n in prepared_model.graph.nodes:
                if n.target in _bn_targets and len(n.args) > 5:
                    new_args    = list(n.args)
                    new_args[5] = False   # training=False
                    n.args      = tuple(new_args)
            prepared_model.recompile()

        # ── Train ─────────────────────────────────────────────────────
        pt2e_utils.move_exported_model_to_train(prepared_model)

        avg_total = AverageMeter()
        avg_curv  = AverageMeter()
        avg_dist  = AverageMeter()
        avg_flag  = AverageMeter()

        p_bar = tqdm.tqdm(
            train_loader,
            desc=f"  Train {epoch+1}/{args.epochs}",
            total=len(train_loader),
        )
        for batch in p_bar:
            img_prev  = batch["img_prev"].to(device)
            img_curr  = batch["img_curr"].to(device)
            d_norm_gt = batch["d_norm"].unsqueeze(1).to(device)
            curv_gt   = batch["curvature"].unsqueeze(1).to(device)
            flag_gt   = batch["flag"].unsqueeze(1).to(device)
            dist_mask = batch["dist_mask"].to(device)

            out = prepared_model(img_prev, img_curr)
            loss, lc, ld, lf = _compute_loss(
                out, d_norm_gt, curv_gt, flag_gt, dist_mask, l1, bce, device
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prepared_model.parameters(), max_norm=10.0)
            optimizer.step()

            bs = img_prev.size(0)
            avg_total.update(loss.item(), bs)
            avg_curv.update(lc, bs)
            avg_dist.update(ld, bs)
            avg_flag.update(lf, bs)
            global_step += 1

            writer.add_scalar("Loss/train_total",     loss.item(), global_step)
            writer.add_scalar("Loss/train_curvature", lc,          global_step)
            writer.add_scalar("Loss/train_distance",  ld,          global_step)
            writer.add_scalar("Loss/train_flag",      lf,          global_step)

            p_bar.set_postfix(
                loss=f"{avg_total.avg:.4f}",
                c=f"{avg_curv.avg:.4f}",
                d=f"{avg_dist.avg:.4f}",
                f=f"{avg_flag.avg:.4f}",
                lr=f"{new_lr:.1e}",
            )

        # Epoch-averaged train losses
        writer.add_scalar("Loss/train_avg_total",     avg_total.avg, epoch + 1)
        writer.add_scalar("Loss/train_avg_curvature", avg_curv.avg,  epoch + 1)
        writer.add_scalar("Loss/train_avg_distance",  avg_dist.avg,  epoch + 1)
        writer.add_scalar("Loss/train_avg_flag",      avg_flag.avg,  epoch + 1)

        # ── Save QAT prepared checkpoint (this epoch + rolling last) ──
        ep_prepared_path = epoch_dir / f"AutoDrive_qat_ep{epoch+1:03d}.pth"
        torch.save(prepared_model.state_dict(), ep_prepared_path)
        torch.save(prepared_model.state_dict(), ckpt_last)
        print(f"\n  Saved QAT prepared  → {ep_prepared_path.name}")

        # ── Convert to INT8 (deepcopy so prepared_model stays intact) ─
        print("  Converting to INT8 (convert_pt2e) …")
        int8_model = convert_pt2e(copy.deepcopy(prepared_model))
        int8_model.to(device)
        int8_size_mb = _model_size_mb(int8_model)
        writer.add_scalar("Info/model_size_int8_MB", int8_size_mb, epoch + 1)

        ep_int8_path = epoch_dir / f"AutoDrive_qat_int8_ep{epoch+1:03d}.pth"
        torch.save(int8_model.state_dict(), ep_int8_path)
        print(f"  Saved INT8 converted → {ep_int8_path.name}  "
              f"({int8_size_mb:.1f} MB vs float {float_size_mb:.1f} MB)")

        # ── Validate prepared model (fake-quantize, float-equivalent) ─
        print("\n  Validating QAT-prepared model (float) …")
        pt2e_utils.move_exported_model_to_eval(prepared_model)
        m_prep = _run_val(prepared_model, val_loader, l1, bce, device, label="prepared")
        _tb_val(writer, "val_prepared", m_prep, epoch + 1)

        # ── Validate INT8 converted model ─────────────────────────────
        print("  Validating INT8-converted model …")
        pt2e_utils.move_exported_model_to_eval(int8_model)
        m_int8 = _run_val(int8_model, val_loader, l1, bce, device, label="int8")
        _tb_val(writer, "val_int8", m_int8, epoch + 1)

        # ── Quantization gap ─────────────────────────────────────────
        _tb_quant_gap(writer, m_prep, m_int8, epoch + 1)

        # ── Console summary ───────────────────────────────────────────
        print(
            f"\n  ┌{'─'*60}┐\n"
            f"  │  Epoch {epoch+1:>3d} validation summary{' '*25}│\n"
            f"  ├{'─'*60}┤\n"
            f"  │  {'Metric':<24} {'Prepared (float)':>16}  {'INT8':>12}  │\n"
            f"  ├{'─'*60}┤\n"
            f"  │  {'Val loss (total)':<24} {m_prep['total']:>16.4f}  {m_int8['total']:>12.4f}  │\n"
            f"  │  {'Curvature loss':<24} {m_prep['curv']:>16.4f}  {m_int8['curv']:>12.4f}  │\n"
            f"  │  {'Distance loss':<24} {m_prep['dist']:>16.4f}  {m_int8['dist']:>12.4f}  │\n"
            f"  │  {'Flag loss':<24} {m_prep['flag']:>16.4f}  {m_int8['flag']:>12.4f}  │\n"
            f"  │  {'Steering MAE (deg)':<24} {m_prep['steer_mae_deg']:>16.2f}  {m_int8['steer_mae_deg']:>12.2f}  │\n"
            f"  │  {'Distance MAE (m)':<24} {m_prep['dist_mae_m']:>16.2f}  {m_int8['dist_mae_m']:>12.2f}  │\n"
            f"  │  {'CIPO flag acc (%)':<24} {m_prep['flag_acc']:>16.2f}  {m_int8['flag_acc']:>12.2f}  │\n"
            f"  └{'─'*60}┘"
        )

        # ── Best model tracking ───────────────────────────────────────
        if m_prep["total"] < best_prepared_val:
            best_prepared_val = m_prep["total"]
            torch.save(prepared_model.state_dict(), ckpt_best_prepared)
            print(f"  ★ New best QAT-prepared val loss: {best_prepared_val:.4f} "
                  f"→ {ckpt_best_prepared.name}")

        if m_int8["total"] < best_int8_val:
            best_int8_val = m_int8["total"]
            torch.save(int8_model.state_dict(), ckpt_best_int8)
            print(f"  ★ New best INT8 val loss:         {best_int8_val:.4f} "
                  f"→ {ckpt_best_int8.name}")

        writer.flush()

        # Return prepared model to train mode for next epoch
        pt2e_utils.move_exported_model_to_train(prepared_model)

    # ─────────────────────────────────────────────────────────────────────
    # Final conversion + save
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  Final INT8 conversion (last epoch weights) …")
    final_int8 = convert_pt2e(copy.deepcopy(prepared_model))
    final_int8.to(device)
    pt2e_utils.move_exported_model_to_eval(final_int8)
    torch.save(final_int8.state_dict(), ckpt_final_int8)
    print(f"  Saved → {ckpt_final_int8}")

    # Final validation
    m_final = _run_val(final_int8, val_loader, l1, bce, device, label="final")
    print(
        f"\n  [Final INT8 Val]  loss {m_final['total']:.4f}  "
        f"steer_mae {m_final['steer_mae_deg']:.2f}°  "
        f"flag_acc {m_final['flag_acc']:.1f}%  "
        f"dist_mae {m_final['dist_mae_m']:.1f} m"
    )
    print(f"\n  Float size : {float_size_mb:.1f} MB")
    print(f"  INT8  size : {_model_size_mb(final_int8):.1f} MB  "
          f"({100*(1 - _model_size_mb(final_int8)/float_size_mb):.0f}% smaller)")

    # ── Optional ONNX export ──────────────────────────────────────────────
    if args.export_onnx:
        onnx_path = ckpt_dir / "AutoDrive_qat_int8.onnx"
        print(f"\n  Exporting INT8 ONNX → {onnx_path}")
        img_prev_ex = torch.randn(1, 3, 512, 1024, device=device)
        img_curr_ex = torch.randn(1, 3, 512, 1024, device=device)
        torch.onnx.export(
            final_int8,
            (img_prev_ex, img_curr_ex),
            str(onnx_path),
            opset_version=17,
            input_names=["image_prev", "image_curr"],
            output_names=["distance", "curvature", "flag_logit"],
            dynamic_axes={
                "image_prev":  {0: "batch"},
                "image_curr":  {0: "batch"},
                "distance":    {0: "batch"},
                "curvature":   {0: "batch"},
                "flag_logit":  {0: "batch"},
            },
        )
        print(f"  Saved ONNX → {onnx_path}")

    writer.flush()
    writer.close()
    print(f"\n{'='*64}")
    print(f"  QAT complete.  Outputs in: {run_dir}")
    print(f"  Use {ckpt_best_int8.name} or {ckpt_final_int8.name}")
    print(f"  for INT8 ONNX export via convert_pytorch_to_onnx.py")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
