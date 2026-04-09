# AutoDrive trainer

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Models.model_components.autodrive.autodrive_network import AutoDrive


class AverageMeter:
    """Tracks running average of a scalar."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


class AutoDriveTrainer:
    """
    Losses
    ------
    distance  : L1( relu_pred, d_norm_gt )     — only when dist_mask=True
                d_norm = (150 - min(d,150)) / 150  → ∈ [0, 1]
    curvature : L1( tanh_pred, curvature_gt )  — always
    flag      : BCEWithLogitsLoss( logits, flag_gt, pos_weight≈0.418 )  — always
                flag_head outputs raw logits; BCEWithLogitsLoss applies sigmoid internally.
                pos_weight < 1 down-weights majority positives (~70.5%) so the
                minority negatives (~29.5%) have fair gradient contribution.
    total     : distance_loss + curvature_loss + flag_loss

    TensorBoard scalars
    -------------------
    Loss/train_*        — per-step raw losses (noisy, every 200 steps)
    Loss/train_avg_*    — epoch-averaged losses via AverageMeter (smooth)
    Loss/val_*          — epoch-averaged validation losses
    Metrics/flag_acc_%  — CIPO flag classification accuracy (0–100)
    Metrics/dist_mae_m  — distance MAE in real metres (CIPO frames only)
    Metrics/lr          — current learning rate
    """

    # positives (~70.5%) are majority → pos_weight < 1 scales their gradient down
    _CIPO_POS_WEIGHT = torch.tensor([0.295 / 0.705])   # ≈ 0.418

    def __init__(self, tensorboard_dir: str = 'runs'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'AutoDriveTrainer — device: {self.device}')

        self.model = AutoDrive().to(self.device)

        self.writer = SummaryWriter(log_dir=tensorboard_dir)

        self.learning_rate = 1e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self._l1  = nn.L1Loss()
        self._bce = nn.BCEWithLogitsLoss(
            pos_weight=self._CIPO_POS_WEIGHT.to(self.device)
        )

        # Per-step state
        self.loss:           torch.Tensor | None = None
        self.loss_distance:  float = 0.0
        self.loss_curvature: float = 0.0
        self.loss_flag:      float = 0.0

        # Epoch-level running averages (reset each epoch)
        self.avg_total     = AverageMeter()
        self.avg_distance  = AverageMeter()
        self.avg_curvature = AverageMeter()
        self.avg_flag      = AverageMeter()

        # Visualization state (updated during run_model)
        self._img_prev_vis:  torch.Tensor | None = None
        self._d_pred_val:    float = 0.0
        self._d_gt_val:      float = 0.0
        self._curv_pred_val: float = 0.0
        self._curv_gt_val:   float = 0.0
        self._flag_pred_val: float = 0.0   # sigmoid(logit) — probability ∈ (0,1)
        self._flag_gt_val:   float = 0.0

    # ------------------------------------------------------------------
    # Epoch management
    # ------------------------------------------------------------------

    def reset_averages(self):
        """Call at the start of each epoch."""
        self.avg_total.reset()
        self.avg_distance.reset()
        self.avg_curvature.reset()
        self.avg_flag.reset()

    # ------------------------------------------------------------------
    # Batch management
    # ------------------------------------------------------------------

    def set_batch(self, batch: dict):
        self.img_prev     = batch["img_prev"].to(self.device)
        self.img_curr     = batch["img_curr"].to(self.device)
        self.d_norm_gt    = batch["d_norm"].unsqueeze(1).to(self.device)
        self.curvature_gt = batch["curvature"].unsqueeze(1).to(self.device)
        self.flag_gt      = batch["flag"].unsqueeze(1).to(self.device)
        self.dist_mask    = batch["dist_mask"].to(self.device)

        self._img_prev_vis = batch["img_prev"][0]
        self._d_gt_val     = batch["d_norm"][0].item()
        self._curv_gt_val  = batch["curvature"][0].item()
        self._flag_gt_val  = batch["flag"][0].item()

    def set_learning_rate(self, lr: float):
        self.learning_rate = lr
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------

    def run_model(self):
        d_pred, curv_pred, flag_logits = self.model(self.img_prev, self.img_curr)

        # Distance loss — only on frames where CIPO is detected with valid distance
        if self.dist_mask.any():
            mask_idx = self.dist_mask.unsqueeze(1)
            loss_d   = self._l1(d_pred[mask_idx], self.d_norm_gt[mask_idx])
        else:
            loss_d = torch.tensor(0.0, device=self.device)

        loss_c = self._l1(curv_pred, self.curvature_gt)
        loss_f = self._bce(flag_logits, self.flag_gt)

        self.loss           = loss_d + loss_c + loss_f
        self.loss_distance  = loss_d.item()
        self.loss_curvature = loss_c.item()
        self.loss_flag      = loss_f.item()

        n = self.img_prev.size(0)
        self.avg_total.update(self.loss.item(), n)
        self.avg_distance.update(self.loss_distance, n)
        self.avg_curvature.update(self.loss_curvature, n)
        self.avg_flag.update(self.loss_flag, n)

        self._d_pred_val    = d_pred[0].item()
        self._curv_pred_val = curv_pred[0].item()
        # Convert logit → probability for visualization
        self._flag_pred_val = torch.sigmoid(flag_logits[0]).item()

    # ------------------------------------------------------------------
    # Validation — returns losses + interpretable metrics
    # ------------------------------------------------------------------

    def validate(self, batch: dict) -> tuple:
        """
        Returns (total_loss, dist_loss, curv_loss, flag_loss,
                 flag_acc_pct, dist_mae_m)

        flag_acc_pct : classification accuracy 0–100 for this batch
        dist_mae_m   : mean absolute error in metres (CIPO frames only)
        """
        self.set_batch(batch)

        d_pred, curv_pred, flag_logits = self.model(self.img_prev, self.img_curr)

        if self.dist_mask.any():
            mask_idx   = self.dist_mask.unsqueeze(1)
            loss_d     = self._l1(d_pred[mask_idx], self.d_norm_gt[mask_idx])
            # d_norm = (150 - d) / 150  →  d_metres = 150 * (1 - d_norm)
            dist_mae_m = (150.0 * (d_pred[mask_idx] - self.d_norm_gt[mask_idx]).abs()).mean().item()
        else:
            loss_d     = torch.tensor(0.0, device=self.device)
            dist_mae_m = 0.0

        loss_c = self._l1(curv_pred, self.curvature_gt)
        loss_f = self._bce(flag_logits, self.flag_gt)
        total  = loss_d + loss_c + loss_f

        # Flag accuracy: logit > 0  ↔  sigmoid > 0.5  → predicted positive
        pred_label = (flag_logits > 0.0).float()
        flag_acc   = (pred_label == self.flag_gt).float().mean().item() * 100.0

        return (total.item(), loss_d.item(), loss_c.item(), loss_f.item(),
                flag_acc, dist_mae_m)

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def loss_backward(self):
        self.loss.backward()

    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_loss(self) -> float:
        return self.loss.item()

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, epoch: int, global_step: int, best_val_loss: float):
        """Save full training state (model + optimizer + counters)."""
        print(f'Saving checkpoint → {path}')
        torch.save({
            'epoch':         epoch,
            'global_step':   global_step,
            'best_val_loss': best_val_loss,
            'model':         self.model.state_dict(),
            'optimizer':     self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> tuple[int, int, float]:
        """
        Load checkpoint and return (start_epoch, global_step, best_val_loss).

        Handles two formats:
          • Full state dict  (saved by save_checkpoint) — restores all counters.
          • Weights-only     (legacy)                   — loads weights only;
                                                          counters reset to 0.
        """
        print(f'Loading checkpoint ← {path}')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict) and 'model' in ckpt:
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch   = ckpt.get('epoch', 0)
            global_step   = ckpt.get('global_step', 0)
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            print(f'  Resuming from epoch {start_epoch + 1}, '
                  f'step {global_step}, best val {best_val_loss:.4f}')
        else:
            self.model.load_state_dict(ckpt)
            start_epoch   = 0
            global_step   = 0
            best_val_loss = float('inf')
            print('  Weights-only checkpoint loaded — counters reset to 0.')

        return start_epoch, global_step, best_val_loss

    def save_model(self, path: str):
        """Save model weights only (lightweight, for inference)."""
        print(f'Saving model weights → {path}')
        torch.save(self.model.state_dict(), path)

    # ------------------------------------------------------------------
    # TensorBoard logging
    # ------------------------------------------------------------------

    def log_train_step(self, step: int):
        """Per-step raw loss (noisy — logged every N steps)."""
        self.writer.add_scalar("Loss/train_total",     self.get_loss(),     step)
        self.writer.add_scalar("Loss/train_distance",  self.loss_distance,  step)
        self.writer.add_scalar("Loss/train_curvature", self.loss_curvature, step)
        self.writer.add_scalar("Loss/train_flag",      self.loss_flag,      step)

    def log_train_epoch(self, epoch: int):
        """Epoch-averaged training losses (smooth — once per epoch)."""
        self.writer.add_scalar("Loss/train_avg_total",     self.avg_total.avg,     epoch)
        self.writer.add_scalar("Loss/train_avg_distance",  self.avg_distance.avg,  epoch)
        self.writer.add_scalar("Loss/train_avg_curvature", self.avg_curvature.avg, epoch)
        self.writer.add_scalar("Loss/train_avg_flag",      self.avg_flag.avg,      epoch)
        self.writer.add_scalar("Metrics/lr",               self.learning_rate,     epoch)

    def log_val_epoch(self, total: float, dist: float, curv: float, flag: float,
                      flag_acc: float, dist_mae_m: float, epoch: int):
        self.writer.add_scalar("Loss/val_total",      total,      epoch)
        self.writer.add_scalar("Loss/val_distance",   dist,       epoch)
        self.writer.add_scalar("Loss/val_curvature",  curv,       epoch)
        self.writer.add_scalar("Loss/val_flag",       flag,       epoch)
        self.writer.add_scalar("Metrics/flag_acc_%",  flag_acc,   epoch)
        self.writer.add_scalar("Metrics/dist_mae_m",  dist_mae_m, epoch)

    def log_test(self, total: float, dist: float, curv: float, flag: float,
                 flag_acc: float, dist_mae_m: float):
        self.writer.add_scalar("Loss/test_total",         total,      0)
        self.writer.add_scalar("Loss/test_distance",      dist,       0)
        self.writer.add_scalar("Loss/test_curvature",     curv,       0)
        self.writer.add_scalar("Loss/test_flag",          flag,       0)
        self.writer.add_scalar("Metrics/test_flag_acc_%", flag_acc,   0)
        self.writer.add_scalar("Metrics/test_dist_mae_m", dist_mae_m, 0)

    def log_train_loss(self, step: int):
        """Kept for backward compatibility."""
        self.log_train_step(step)

    def save_visualization(self, step: int):
        if self._img_prev_vis is None:
            return

        d_pred_m = 150.0 * (1.0 - self._d_pred_val)
        d_gt_m   = 150.0 * (1.0 - self._d_gt_val)

        img_vis = self._img_prev_vis.permute(1, 2, 0).numpy()
        img_vis = (img_vis * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        ax.imshow(img_vis)
        txt = (
            f"dist pred: {d_pred_m:.1f} m  |  dist GT: {d_gt_m:.1f} m\n"
            f"curv pred: {self._curv_pred_val:.5f}  |  curv GT: {self._curv_gt_val:.5f}\n"
            f"flag prob: {self._flag_pred_val:.2f}  |  flag GT: {self._flag_gt_val:.0f}"
        )
        ax.text(10, 30, txt, color="black", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="black",
                          boxstyle="round,pad=0.4"))
        self.writer.add_figure("Visualization/sample", fig, global_step=step)
        plt.close(fig)

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('AutoDriveTrainer: finished.')
