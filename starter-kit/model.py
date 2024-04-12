import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy, MeanAbsoluteError
from transformers import AutoConfig, AutoModelForImageClassification


class EarthQuakeModel(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        num_classes = 1

        config = AutoConfig.from_pretrained(self.hparams["model_name"])
        config.num_channels = self.hparams["in_chans"]
        config.num_labels = num_classes
        self.model = AutoModelForImageClassification.from_config(config)

        self.accuracy = Accuracy("multiclass", num_classes=2)
        self.regr_metric = MeanAbsoluteError()

        self.regression_loss = nn.MSELoss()
        self.train_transform = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        if hasattr(x, "logits"):
            x = x.logits
        return x.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams["lr"], weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=self.hparams["lr"],
            pct_start=0.1,
            cycle_momentum=False,
            div_factor=1e9,
            final_div_factor=1e4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        sample, label, mag = (batch["sample"], batch["label"], batch["magnitude"])

        sample = self.train_transform(sample)
        y_r = self(sample)

        loss = self.regression_loss(y_r, mag)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        sample, label, mag = (batch["sample"], batch["label"], batch["magnitude"])

        y_r = self(sample)

        loss = self.regression_loss(y_r, mag)

        self.accuracy((y_r >= 1).to(torch.int), label)
        self.log("val_acc", self.accuracy)
        self.regr_metric(y_r, mag)
        self.log(f"val_{self.regr_metric.__class__.__name__}", self.regr_metric)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        sample, label, mag = (batch["sample"], batch["label"], batch["magnitude"])

        y_r = self(sample)

        self.accuracy((y_r >= 1).to(torch.int), label)
        self.log("val_acc", self.accuracy)
        self.regr_metric(y_r, mag)
        self.log(f"val_{self.regr_metric.__class__.__name__}", self.regr_metric)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        sample = batch["sample"]
        y_r = self(sample)
        return y_r
