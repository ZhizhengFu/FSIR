import torch
import torch.nn as nn
from torch.fft import fft2, ifft2


def frequency_downsample(x: torch.Tensor, scale_factor: int):
    x = torch.stack(x.chunk(scale_factor, dim=2), 4)
    x = torch.cat(x.chunk(scale_factor, dim=3), 4)
    return x.mean(-1).repeat(1, 1, scale_factor, scale_factor)


class TVRestorer(nn.Module):
    def __init__(self, num_iters: int = 50):
        super().__init__()
        self.num_iters = num_iters

    def forward(self, STy, Fk, sf, alpha=1e-6, gam=0.1):
        FSTy = fft2(STy)
        Fr = Fk.conj() * FSTy / (frequency_downsample(Fk.abs().pow(2), sf) + alpha)
        predicted_image = ifft2(Fr).real

        DH = torch.zeros_like(predicted_image)
        DV = torch.zeros_like(predicted_image)
        DH[..., 0, 0], DH[..., 0, 1] = 1, -1
        DV[..., 0, 0], DV[..., 1, 0] = 1, -1
        FDV, FDH = fft2(DV), fft2(DH)
        FDVConj, FDHConj = FDV.conj(), FDH.conj()
        FD2 = FDV.abs().pow(2) + FDH.abs().pow(2) + 1e-8

        U1, U2 = predicted_image.clone(), predicted_image.clone()
        D1, D2 = torch.zeros_like(predicted_image), torch.zeros_like(predicted_image)

        x_old = predicted_image.clone()
        t_old = 1.0

        for k_iter in range(self.num_iters):
            gam_k = gam * 0.5 ** (k_iter / self.num_iters)

            Fr = Fk.conj() * FSTy + alpha * (
                FDHConj * fft2(U1 - D1) + FDVConj * fft2(U2 - D2)
            )
            Fr = (
                Fr
                - Fk.conj()
                * frequency_downsample(Fk * Fr / FD2, sf)
                / (frequency_downsample(Fk.abs().pow(2) / FD2, sf) + alpha)
            ) / (alpha * FD2)

            predicted_image_new = ifft2(Fr).real

            t_new = (1 + (1 + 4 * t_old**2) ** 0.5) / 2
            predicted_image = predicted_image_new + ((t_old - 1) / t_new) * (
                predicted_image_new - x_old
            )
            x_old = predicted_image_new.clone()
            t_old = t_new

            DHr, DVr = ifft2(Fr * FDH).real, ifft2(Fr * FDV).real
            NU1, NU2 = DHr + D1, DVr + D2
            NU = torch.sqrt(NU1.pow(2) + NU2.pow(2) + 1e-8)
            A = torch.maximum(torch.zeros_like(NU), NU - gam_k) + 1e-8
            A = A / (A + gam_k)
            U1, U2 = A * NU1, A * NU2
            D1, D2 = D1 + (DHr - U1), D2 + (DVr - U2)

        return predicted_image


class HyperNet(nn.Module):
    def __init__(self, in_nc=2, channel=64, out_nc=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x) + 1e-7


class FSIRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_net = TVRestorer()

    def forward(self, Fk, y, sigma, sf):
        STy = torch.zeros(
            (y.shape[0], 3, y.shape[2] * sf, y.shape[3] * sf), device=y.device
        )
        STy[..., ::sf, ::sf] = y
        x = self.init_net(STy, Fk, sf)
        return x
