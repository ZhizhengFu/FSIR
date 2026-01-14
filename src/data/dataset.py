import torch
import random
from pathlib import Path
import torch.nn.functional as F
from torch.fft import fft2, ifft2
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision import transforms as T
from src.config import Config
from src.utils import KernelSynthesizer


class FSIRDataset(Dataset):
    def __init__(self, opt: Config, mode: str):
        self.opt = opt
        self.mode = mode
        self.transform = self._get_transform()
        self.k_synthesizer = KernelSynthesizer() if mode == "train" else None
        self.img_path = list(Path(opt.img_path).glob("*"))
        self.kernel_list = torch.load(opt.kernel_path) if opt.kernel_path else None

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = decode_image(self.img_path[idx], ImageReadMode.RGB) / 255
        return self.transform(img) if self.transform else img

    def _get_transform(self) -> T.Compose | None:
        if self.mode != "train":
            return None
        return T.Compose(
            [
                T.RandomCrop(self.opt.patch_size),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
            ]
        )

    def collate_fn(self, batch):
        x = torch.stack(batch)
        batch_size = len(batch)

        if self.mode == "train":
            sf = random.choice(self.opt.sf_list)
            sigma = torch.empty(batch_size, 1, 1, 1).uniform_(0, self.opt.sigma_max)
            k_type = random.choice(["gaussian", "motion"])
            kernel_func = getattr(self.k_synthesizer, f"gen_{k_type}_kernel")
            kernel = torch.stack([kernel_func() for _ in range(batch_size)])[None]
        else:
            sf = self.opt.sf
            sigma = torch.full((batch_size, 1, 1, 1), self.opt.sigma)
            kernel = (
                self.kernel_list[self.opt.k_idx].expand(batch_size, 1, -1, -1)
                if self.kernel_list
                else self.opt.kernel.expand(batch_size, 1, -1, -1)
            )

        return self.image_degradation(x, kernel, sf, sigma)

    @staticmethod
    def image_degradation(x, kernel, sf, sigma):
        """
        y = skx+n

        :param x: Description
        :param kernel: Description
        :param sf: Description
        :param sigma: Description
        """
        H, W = x.shape[-2:]
        H = H // sf * sf
        W = W // sf * sf
        x = x[..., :H, :W]
        kh, kw = kernel.shape[-2:]
        padding = (0, W - kw, 0, H - kh)
        kernel = F.pad(kernel, padding).double()
        shifts = (-(kh // 2), -(kw // 2))
        Fk = fft2(kernel.roll(shifts, (-2, -1)))
        kx = ifft2(fft2(x) * Fk).real
        skx = kx[..., ::sf, ::sf]
        y = skx + torch.randn_like(skx) * sigma
        return x, Fk, y, sigma, sf
