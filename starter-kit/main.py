import time

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from model import EarthQuakeModel
from omegaconf import DictConfig
from torchgeo.datamodules import QuakeSetDataModule


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(args: DictConfig):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    data_module = QuakeSetDataModule(**args.dataset)
    model = EarthQuakeModel(**args.model)

    experiment_id = time.strftime("%Y%m%d-%H%M%S")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{experiment_id}",
        filename="earthquake-detection-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        **args.trainer,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=50,
        precision="32-true",
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
