from __future__ import annotations
import os
import random
import pytorch_lightning as pl

from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)


from pytorch_lightning.loggers import WandbLogger
import wandb

import data_module


from data_module import SceneDataModule

from model import DesignStyleModel

CONFIG = {
    "data_dir": r"C:\Users\Dell\.cache\kagglehub\competitions\cse-281-spring-26-scene-style-classification\StyleClassificationIndoors\StyleClassificationIndoors\train",
    "num_workers": 2,
    "batch_size": 4,
    "mixup_alpha": 0.0,

    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std":  [0.229, 0.224, 0.225],

    "model_name":"resnet50",
    "pretrained_backbone": True,
    "pretrained_path":     None,
    "num_classes":         17,

    "texture_weight":  0.0,
    "label_smoothing": 0.05,

    "backbone_lr":      1e-5,
    "head_lr":          4e-4,

    "freeze_backbone":  True,
    "unfreeze_epoch":   1,

    "dropout":          0.3,

    "max_epochs":  15,
    "precision":   "16-mixed",
    "es_patience": 4,
    "val_split": 0.15,

    "project":  "cse281-scene-classification",
    "run_name": "run04-resnet50-baseline",    
}

EXPERIMENTS = [
    # {
    #     "model_name": "convnext_tiny",
    #     "run_name": "seq01-convnext-tiny",
    #     "backbone_lr": 1e-5,
    #     "head_lr": 4e-4,
    #     "dropout": 0.30,
    #     "unfreeze_epoch": 7,
    #     "label_smoothing": 0.05,
    #     "texture_weight": 0.0,
    # },
    # {
    #     "model_name": "resnet50",
    #     "run_name": "resnet50-256-deeper-head",
    #     "backbone_lr": 1e-5,
    #     "head_lr": 3e-4,
    #     "dropout": 0.30,
    #     "unfreeze_epoch": 5,
    #     "label_smoothing": 0.02,
    #     "texture_weight": 0.0,
    # },
    # {
    #     "model_name": "convnext_small",
    #     "run_name": "seq05-convnext-small-stronger-ft",
    #     "backbone_lr": 1e-5,
    #     "head_lr": 2e-4,
    #     "dropout": 0.30,
    #     "unfreeze_epoch": 6,
    #     "label_smoothing": 0.05,
    #     "texture_weight": 0.0,
    # },
    # {
    #     "model_name": "convnext_base",
    #     "run_name": "convnext-base-22k-match-56pct-style",
    #     "backbone_lr": 5e-5,
    #     "head_lr": 3e-4,
    #     "dropout": 0.30,
    #     "unfreeze_epoch": 0,
    #     "label_smoothing": 0.10,
    #     "texture_weight": 0.0,
    # },
    # {
    #     "model_name": "efficientnet_b4",
    #     "run_name": "seq04-efficientnet-b4",
    #     "backbone_lr": 5e-6,
    #     "head_lr": 3e-4,
    #     "dropout": 0.40,
    #     "unfreeze_epoch": 8,
    #     "label_smoothing": 0.05,
    #     "texture_weight": 0.0,
    # },
    # {
    # "model_name": "swin_tiny",
    # "run_name": "swin-tiny-256-deeper-head",
    # "backbone_lr": 6e-6,
    # "head_lr": 2e-4,
    # "dropout": 0.30,
    # "unfreeze_epoch": 7,
    # "label_smoothing": 0.05,
    # "texture_weight": 0.0,
    # },
    {
        "model_name": "swin_small",
        "run_name": "swin-small-22k-256",
        "backbone_lr": 5e-5,
        "head_lr": 3e-4,
        "dropout": 0.30,
        "unfreeze_epoch": 0,
        "label_smoothing": 0.1,
        "texture_weight": 0.0,
    },
]


class UnfreezeBackboneCallback(pl.Callback):

    def __init__(self, unfreeze_epoch: int = 5):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self._unfrozen = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self._unfrozen and trainer.current_epoch >= self.unfreeze_epoch:
            print(f"\n[train.py] ✓ Unfreezing backbone at epoch {trainer.current_epoch}.")
            pl_module.unfreeze_backbone()
            self._unfrozen = True


def build_callbacks(config: dict) -> list:
    return [
        EarlyStopping(
            monitor="val_acc",
            patience=config["es_patience"],
            mode="max",
            verbose=True,
        ),
        ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
        filename="best-{epoch:02d}-{val_acc:.4f}",
        verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        UnfreezeBackboneCallback(unfreeze_epoch=config["unfreeze_epoch"]),
        RichProgressBar(),
    ]


def apply_imagenet_normalization(config: dict) -> None:

    if hasattr(data_module, "CUSTOM_MEAN"):
        data_module.CUSTOM_MEAN = config["normalization_mean"]
        print(f"[train.py] data_module.CUSTOM_MEAN → {data_module.CUSTOM_MEAN}")

    if hasattr(data_module, "CUSTOM_STD"):
        data_module.CUSTOM_STD = config["normalization_std"]
        print(f"[train.py] data_module.CUSTOM_STD  → {data_module.CUSTOM_STD}")


def infer_num_classes(datamodule: SceneDataModule, fallback: int) -> int:
    if hasattr(datamodule, "num_classes") and datamodule.num_classes is not None:
        return datamodule.num_classes

    print(f"[train.py] Could not infer num_classes; using fallback={fallback}")
    return fallback


def log_class_names(datamodule: SceneDataModule, wandb_logger: WandbLogger) -> None:
    if hasattr(datamodule, "class_names") and datamodule.class_names is not None:
        names = datamodule.class_names
        wandb_logger.experiment.config.update({"class_names": names})
        print(f"[train.py] Classes ({len(names)}): {names}")


def run_single_experiment(config: dict):
    global_seed = random.randint(0, 9999)
    print(f"\n[train.py] Global seed: {global_seed}")
    pl.seed_everything(global_seed, workers=True)

    apply_imagenet_normalization(config)

    datamodule = SceneDataModule(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=global_seed,
        mixup_alpha=config["mixup_alpha"],
        val_split=config.get("val_split", 0.01),
    )
    datamodule.setup()

    config["num_classes"] = infer_num_classes(datamodule, config["num_classes"])
    print(f"[train.py] Number of classes: {config['num_classes']}")

    model = DesignStyleModel(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        texture_weight=config["texture_weight"],
        label_smoothing=config["label_smoothing"],
        backbone_lr=config["backbone_lr"],
        head_lr=config["head_lr"],
        pretrained_backbone=config["pretrained_backbone"],
        pretrained_path=config["pretrained_path"],
        freeze_backbone=config["freeze_backbone"],
        dropout=config["dropout"],
    )

    wandb_logger = WandbLogger(
        project=config["project"],
        name=config["run_name"],
        log_model=False,
    )
    wandb_logger.log_hyperparams({**config, "seed": global_seed})
    log_class_names(datamodule, wandb_logger)

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        precision=config["precision"],
        accelerator="gpu",
        devices=1,
        callbacks=build_callbacks(config),
        logger=wandb_logger,
        deterministic=False,         
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,       
        log_every_n_steps=10,
        accumulate_grad_batches=16,
    )

    print("\n" + "=" * 70)
    print(f" Run: {config['run_name']}")
    print(f" Backbone: {config['model_name']} (pretrained={config['pretrained_backbone']})")
    print(f" Phase 1: head-only for {config['unfreeze_epoch']} epochs")
    print(f" Phase 2: full fine-tune for remaining epochs")
    print("=" * 70 + "\n")

    trainer.fit(model, datamodule=datamodule)

    print("\n" + "=" * 70)
    print(" Testing best checkpoint on held-out test set")
    print("=" * 70 + "\n")

    if datamodule.test_ds is not None:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
    else:
        print("[train.py] No internal test set for this run. Skipping trainer.test().")

    print(f"\n[train.py] ✓ Run complete. Results logged to W&B project: {config['project']}")


def main():
    import copy
    import gc
    import torch

    for experiment in EXPERIMENTS:
        run_config = copy.deepcopy(CONFIG)
        run_config.update(experiment)

        print("\n" + "#" * 80)
        print(f"STARTING EXPERIMENT: {run_config['run_name']}")
        print(f"MODEL: {run_config['model_name']}")
        print("#" * 80 + "\n")

        run_single_experiment(run_config)
        import wandb
        wandb.finish()

        print("\n" + "#" * 80)
        print(f"FINISHED EXPERIMENT: {run_config['run_name']}")
        print("#" * 80 + "\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()