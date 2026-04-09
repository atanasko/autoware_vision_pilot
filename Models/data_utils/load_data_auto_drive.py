"""
AutoDrive ZOD dataset loader.

Labels:  {zod_root}/labels/{seq}/*.json  (one per frame, ISO-timestamp filename)
Images:  {zod_root}/images_blur_*/sequences/{seq}/camera_front_blur/

Sequential pairs (T-1, T) within each sequence only — no cross-sequence pairing.
85 / 10 / 5 split at sequence level to avoid temporal leakage.

Distance GT:
    d_norm = (150 - min(d, 150)) / 150  →  ∈ [0, 1]
    When cipo_detected=False, d_norm=0.0 and dist_mask=False
    so the trainer skips the distance loss for that sample.

Augmentations (training only):
    Horizontal flip (p=0.5):
        Both frames flipped left-right.
        Curvature sign negated  (a right curve becomes a left curve after flip).
        Distance and flag are unchanged.
    Colour / noise applied identically to both frames via albumentations ReplayCompose.
"""

import json
import random
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from Models.data_parsing.AutoDrive.zod.zod_utils import get_images_blur_dir

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_NET_W, _NET_H = 1024, 512
_D_MAX         = 150.0

# Colour / noise augmentation applied identically to both frames
_COLOUR_AUG = A.ReplayCompose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
    A.GaussNoise(noise_scale_factor=0.2, p=0.3),
    A.ISONoise(color_shift=(0.05, 0.2), intensity=(0.1, 0.3), p=0.2),
    A.ToGray(num_output_channels=3, method='weighted_average', p=0.05),
])

_RESIZE = A.Compose([A.Resize(width=_NET_W, height=_NET_H)])


def _norm_distance(d_metres: float) -> float:
    return (_D_MAX - min(d_metres, _D_MAX)) / _D_MAX


def _to_tensor(img_np: np.ndarray) -> torch.Tensor:
    img = TF.to_tensor(img_np)
    img = TF.normalize(img, _IMAGENET_MEAN, _IMAGENET_STD)
    return img


def _augment_pair(img_prev: np.ndarray, img_curr: np.ndarray,
                  curvature: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Apply training augmentations to (prev, curr) frame pair.

    Horizontal flip (p=0.5):
        Both frames flipped.  Curvature sign negated.

    Colour / noise:
        Same random parameters applied to both frames so the temporal
        relationship is preserved.
    """
    # Horizontal flip — negate curvature
    if random.random() < 0.5:
        img_prev  = np.ascontiguousarray(img_prev[:, ::-1, :])
        img_curr  = np.ascontiguousarray(img_curr[:, ::-1, :])
        curvature = -curvature

    # Colour / noise — same parameters on both frames via ReplayCompose
    result    = _COLOUR_AUG(image=img_prev)
    img_prev  = result["image"]
    img_curr  = A.ReplayCompose.replay(result["replay"], image=img_curr)["image"]

    return img_prev, img_curr, curvature


class AutoDriveDataset(Dataset):
    """
    Each __getitem__ returns a dict:
        img_prev    : (3, H, W) float tensor
        img_curr    : (3, H, W) float tensor
        d_norm      : scalar float  ∈ [0, 1]
        curvature   : scalar float  (1/m, sign-correct after flip)
        flag        : scalar float  {0.0, 1.0}
        dist_mask   : bool (True = distance loss active)
    """

    def __init__(self, zod_root: str | Path, sequences: list[str], is_train: bool = True):
        self.is_train = is_train
        self.pairs: list[tuple] = []

        zod_root = Path(zod_root)
        for seq in sequences:
            label_dir = zod_root / "labels" / seq
            if not label_dir.exists():
                continue

            label_files = sorted(label_dir.glob("*.json"))
            if len(label_files) < 2:
                continue

            img_dir = get_images_blur_dir(zod_root, seq)
            records = []
            for lf in label_files:
                with open(lf) as fh:
                    rec = json.load(fh)
                img_path = img_dir / rec["image"]
                if img_path.exists():
                    records.append((str(img_path), rec))

            for i in range(1, len(records)):
                path_prev, _        = records[i - 1]
                path_curr, lbl_curr = records[i]

                cipo      = bool(lbl_curr.get("cipo_detected", False))
                raw_dist  = lbl_curr.get("distance_to_in_path_object")
                curvature = float(lbl_curr.get("curvature") or 0.0)

                if cipo and raw_dist is not None:
                    d_norm    = _norm_distance(float(raw_dist))
                    dist_mask = True
                else:
                    d_norm    = 0.0
                    dist_mask = False

                flag = 1.0 if cipo else 0.0
                self.pairs.append((path_prev, path_curr, d_norm, curvature, flag, dist_mask))

        print(f"AutoDriveDataset ({'train' if is_train else 'val/test'}): "
              f"{len(self.pairs):,} pairs from {len(sequences)} sequences.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        path_prev, path_curr, d_norm, curvature, flag, dist_mask = self.pairs[idx]

        img_prev = np.array(Image.open(path_prev).convert("RGB"))
        img_curr = np.array(Image.open(path_curr).convert("RGB"))

        # Resize to network input size
        img_prev = _RESIZE(image=img_prev)["image"]
        img_curr = _RESIZE(image=img_curr)["image"]

        if self.is_train:
            img_prev, img_curr, curvature = _augment_pair(img_prev, img_curr, curvature)

        return {
            "img_prev":  _to_tensor(img_prev),
            "img_curr":  _to_tensor(img_curr),
            "d_norm":    torch.tensor(d_norm,    dtype=torch.float32),
            "curvature": torch.tensor(curvature, dtype=torch.float32),
            "flag":      torch.tensor(flag,      dtype=torch.float32),
            "dist_mask": torch.tensor(dist_mask, dtype=torch.bool),
        }


class LoadDataAutoDrive:
    """
    Splits all ZOD sequences 85 / 10 / 5 at sequence level.

        data = LoadDataAutoDrive("/path/to/zod")
        data.train / data.val / data.test  →  AutoDriveDataset
    """

    TRAIN_FRAC = 0.85
    VAL_FRAC   = 0.10

    def __init__(self, zod_root: str | Path):
        zod_root   = Path(zod_root)
        labels_dir = zod_root / "labels"

        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        all_seqs = sorted([d.name for d in labels_dir.iterdir() if d.is_dir()])
        if not all_seqs:
            raise FileNotFoundError(f"No sequence folders found under {labels_dir}")

        n       = len(all_seqs)
        n_train = max(1, round(n * self.TRAIN_FRAC))
        n_val   = max(1, round(n * self.VAL_FRAC))

        train_seqs = all_seqs[:n_train]
        val_seqs   = all_seqs[n_train : n_train + n_val]
        test_seqs  = all_seqs[n_train + n_val :]

        print(f"Sequences — train: {len(train_seqs)}  val: {len(val_seqs)}  test: {len(test_seqs)}")

        self.train = AutoDriveDataset(zod_root, train_seqs, is_train=True)
        self.val   = AutoDriveDataset(zod_root, val_seqs,   is_train=False)
        self.test  = AutoDriveDataset(zod_root, test_seqs,  is_train=False)
