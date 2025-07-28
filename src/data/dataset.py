import torch
import random
from pathlib import Path
import torch.nn.functional as F
from torch.fft import fft2, ifft2
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms as T
from src.config import Config
from src.utils import KernelSynthesizer


class FSIRDataset(Dataset):
    def __init__(self, opt: Config, mode: str):
        self.opt = opt
        self.mode = mode
        self.transform = self._get_transform()
        self.k_synthesizer = KernelSynthesizer()
        self.img_paths = list(Path(opt.root_dir).glob("*"))
        self.val_kernel = torch.load("src/kernels/kernels_12.pt")[0]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = read_image(self.img_paths[idx]) / 255
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
        X = torch.stack(batch)
        batch_size = len(batch)

        if self.mode == "train":
            sf = random.choice(self.opt.sf)
            k_type = random.choice(["gaussian", "motion"])
            kernel_func = getattr(self.k_synthesizer, f"gen_{k_type}_kernel")
            K = torch.stack([kernel_func() for _ in range(batch_size)]).unsqueeze(1)
            sigma = torch.empty(batch_size, 1, 1, 1).uniform_(0, self.opt.sigma_max)
        elif self.mode == "val":
            sf = 3
            K = self.val_kernel.expand(batch_size, 1, -1, -1)
            sigma = torch.zeros(batch_size, 1, 1, 1)
        else:
            sf = self.opt.sf
            K = self.opt.kernel
            sigma = torch.full((batch_size, 1, 1, 1), self.opt.sigma)

        H, W = X.shape[-2:]
        H_r, W_r = H % sf, W % sf
        X = X[..., : H - H_r, : W - W_r]
        return self._process_batch(X, K, sigma, sf)

    @staticmethod
    def _process_batch(X, K, sigma, sf):
        psf_h, psf_w = K.shape[-2:]
        target_h, target_w = X.shape[-2:]
        padding = (0, target_w - psf_w, 0, target_h - psf_h)
        otf = F.pad(K, padding).double()
        shifts = (-(psf_h // 2), -(psf_w // 2))
        FK = fft2(otf.roll(shifts, (-2, -1)))
        KX = ifft2(fft2(X) * FK).real
        SKX = KX[..., ::sf, ::sf]
        SKX_n = SKX + torch.randn_like(SKX) * sigma
        return X, FK, SKX_n, sigma, sf
