import torch
import torch.nn as nn
import lightning as L
from pathlib import Path
from src.model import FSIRNet
from src.config import Config
from torchvision.utils import save_image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class FSIR(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = FSIRNet()
        self.psnr = PeakSignalNoiseRatio(data_range=(0, 1))
        self.ssim = StructuralSimilarityIndexMeasure()
        self.mse = nn.MSELoss()
        self.curr_sf: int = 0
        self.k_idx: int = 0

    def training_step(self, batch, batch_idx):
        x, Fk, y, sigma, sf = batch
        x_est = self.model(Fk, y, sigma, sf)

        loss = self.mse(x, x_est)
        batch_size = x.size(0)

        self.log("train_mse_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("train_mean_grad_norm", self.log_gradient_norms(), prog_bar=True)
        self.log("train_lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)  # type: ignore

        return loss

    def validation_step(self, batch, batch_idx):
        x, Fk, y, sigma, sf = batch
        x_est = self.model(Fk, y, sigma, sf)

        mse_loss = self.mse(x, x_est)
        psnr_value = self.psnr(x, x_est)
        ssim_value = self.ssim(x, x_est)
        batch_size = x.size(0)

        self.log("val_mse_loss", mse_loss, prog_bar=True, batch_size=batch_size)
        self.log("val_psnr", psnr_value, prog_bar=True, batch_size=batch_size)
        self.log("val_ssim", ssim_value, prog_bar=True, batch_size=batch_size)

        save_dir = (
            Path(self.logger.save_dir)  # type: ignore
            / self.logger.name  # type: ignore
            / f"version_{self.logger.version}"  # type: ignore
            / "sr_images"
            / str(batch_idx)
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{self.global_step}.png"
        save_image(x_est, save_path)

        return mse_loss

    def test_step(self, batch, batch_index):
        x, Fk, y, sigma, sf = batch
        x_est = self.model(Fk, y, sigma, sf)

        mse_loss = self.mse(x, x_est)
        psnr_value = self.psnr(x, x_est)
        ssim_value = self.ssim(x, x_est)
        batch_size = x.size(0)

        self.log("test_mse_loss", mse_loss, batch_size=batch_size)
        self.log("test_psnr", psnr_value, batch_size=batch_size)
        self.log("test_ssim", ssim_value, batch_size=batch_size)

        if self.curr_sf!=0 and self.curr_sf == sf:
            self.k_idx += 1
        else:
            self.k_idx = 1
        self.curr_sf = sf
        save_dir = (
            Path(self.logger.save_dir)  # type: ignore
            / self.logger.name  # type: ignore
            / f"version_{self.logger.version}"  # type: ignore
            / "test_images"
            / f"sigma_{sigma.item()}"
            / f"sf_{sf}-k_{self.k_idx}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        save_image(x_est, save_dir / f"sr_{batch_index}.png")
        save_image(x, save_dir / f"hr_{batch_index}.png")

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
        x, Fk, y, sigma, sf = batch
        batch = (
            x.to(self.device),
            Fk.to(self.device),
            y.to(self.device),
            sigma.to(self.device),
            sf,
        )
        return batch
