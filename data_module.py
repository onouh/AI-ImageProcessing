import os
import random

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

CUSTOM_MEAN = [0.485, 0.456, 0.406]
CUSTOM_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224

def get_transforms(stage: str):
    if stage == "train":
        return v2.Compose([
            v2.ToImage(),

            v2.RandomResizedCrop(
                size=(IMAGE_SIZE, IMAGE_SIZE),
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
                antialias=True,
            ),

            v2.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05,
            ),

            v2.RandomRotation(degrees=10),
            v2.RandomHorizontalFlip(p=0.5),

            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD),
        ])

    return v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD),
    ])

class MixUpCollate:
    """
    Optional hard-label MixUp.
    For now, keep alpha=0.0 in train.py for clean baseline.
    """

    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

        if self.alpha <= 0.0:
            return images, labels

        lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample())
        idx = torch.randperm(images.size(0))

        mixed_images = lam * images + (1.0 - lam) * images[idx]
        mixed_labels = labels if lam >= 0.5 else labels[idx]

        return mixed_images, mixed_labels


def plain_collate(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


class SceneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        mixup_alpha: float = 0.0,
        val_split: float = 0.01,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.mixup_alpha = mixup_alpha
        self.val_split = val_split

        self.class_names = None
        self.num_classes = None
        
    def setup(self, stage=None):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        base_dataset = ImageFolder(root=self.data_dir)

        self.class_names = base_dataset.classes
        self.num_classes = len(base_dataset.classes)

        print(f"[data_module.py] Found {len(base_dataset)} images.")
        print(f"[data_module.py] Found {self.num_classes} classes.")
        print(f"[data_module.py] Classes: {self.class_names}")

        targets = base_dataset.targets

        rng = random.Random(self.seed)

        train_indices = []
        val_indices = []
        test_indices = []

        for class_id in range(self.num_classes):
            class_indices = [
                i for i, target in enumerate(targets)
                if target == class_id
            ]

            rng.shuffle(class_indices)

            n = len(class_indices)
            n_val = max(1, int(self.val_split * n))

            val_indices.extend(class_indices[:n_val])
            train_indices.extend(class_indices[n_val:])

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

        train_dataset = ImageFolder(
            root=self.data_dir,
            transform=get_transforms("train"),
        )

        val_dataset = ImageFolder(
            root=self.data_dir,
            transform=get_transforms("val"),
        )

        self.train_ds = Subset(train_dataset, train_indices)
        self.val_ds = Subset(val_dataset, val_indices)
        self.test_ds = None

        print(f"[data_module.py] Train size: {len(self.train_ds)}")
        print(f"[data_module.py] Val size:   {len(self.val_ds)}")
        print(f"[data_module.py] Test size:  0")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            collate_fn=MixUpCollate(alpha=self.mixup_alpha),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            collate_fn=plain_collate,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None

        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            collate_fn=plain_collate,
        )