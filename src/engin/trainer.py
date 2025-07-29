import torch
import torch.nn as nn
import lightning as L
from pathlib import Path
from src.model import FSIRNet
from src.config import Config
from torchvision.utils import save_image
from torchmetrics.audio import SignalNoiseRatio
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class FSIR(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = FSIRNet()
        self.snr = SignalNoiseRatio()
        self.psnr = PeakSignalNoiseRatio(data_range=(0, 1))
        self.ssim = StructuralSimilarityIndexMeasure()
        self.mse = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        X, FK, SKX_n, sigma, sf = batch
        output = self.model(FK, SKX_n, sigma, sf)

        loss = self.mse(X, output)
        batch_size = X.size(0)

        self.log("train_mse_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("train_mean_grad_norm", self.log_gradient_norms(), prog_bar=True)
        self.log("train_lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)  # type: ignore

        return loss

    def validation_step(self, batch, batch_idx):
        X, FK, SKX_n, sigma, sf = batch
        output = self.model(FK, SKX_n, sigma, sf)

        mse_loss = self.mse(X, output)
        psnr_value = self.psnr(X, output)
        ssim_value = self.ssim(X, output)
        snr_value = self.snr(X, output)
        batch_size = X.size(0)

        self.log("val_mse_loss", mse_loss, prog_bar=True, batch_size=batch_size)
        self.log("val_psnr", psnr_value, prog_bar=True, batch_size=batch_size)
        self.log("val_ssim", ssim_value, prog_bar=True, batch_size=batch_size)
        self.log("val_snr", snr_value, prog_bar=True, batch_size=batch_size)

        save_dir = (
            Path(self.logger.save_dir)  # type: ignore
            / self.logger.name  # type: ignore
            / f"version_{self.logger.version}"  # type: ignore
            / "sr_images"
            / str(batch_idx)
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{self.global_step}.png"
        save_image(output, save_path)

        return mse_loss

    def test_step(self, batch, batch_index):
        X, FK, SKX_n, sigma, sf = batch
        output = self.model(FK, SKX_n, sigma, sf)

        mse_loss = self.mse(X, output)
        psnr_value = self.psnr(X, output)
        ssim_value = self.ssim(X, output)
        snr_value = self.snr(X, output)
        batch_size = X.size(0)

        self.log("test_mse_loss", mse_loss, batch_size=batch_size)
        self.log("test_psnr", psnr_value, batch_size=batch_size)
        self.log("test_ssim", ssim_value, batch_size=batch_size)
        self.log("test_snr", snr_value, batch_size=batch_size)

        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.config.step_size, gamma=self.config.gamma
        )
        return [optimizer], [scheduler]

    def log_gradient_norms(self):
        total_grad_norm = 0.0
        num_params = 0
        for param in self.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item()
                num_params += 1
        if num_params == 0:
            return 0
        return total_grad_norm / num_params

    def on_before_batch_transfer(self, batch, dataloader_idx):
        X, FK, SKX_n, sigma, sf = batch
        batch = (
            X.to(self.device),
            FK.to(self.device),
            SKX_n.to(self.device),
            sigma.to(self.device),
            sf,
        )
        return batch
