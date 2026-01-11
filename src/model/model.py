import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2
from .backbone import ResUNet


def splits_and_mean(a, sf):
    b = torch.stack(a.chunk(sf, 2), 4)
    b = torch.cat(b.chunk(sf, 3), 4)
    return b.mean(-1)


class TVNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        SKX_n,
        FK,
        FCK,
        F2K,
        FCKFKy,
        sf,
        alpha,
        maxiter=2,
    ):
        X = F.interpolate(SKX_n, scale_factor=sf, mode="nearest")
        DH, DV = torch.zeros_like(X), torch.zeros_like(X)
        DH[..., 0, 0], DH[..., 0, 1] = 1, -1
        DV[..., 0, 0], DV[..., 1, 0] = 1, -1
        FDV, FDH = fft2(DV), fft2(DH)
        FDVC, FDHC = FDV.conj(), FDH.conj()
        F2DV, F2DH = FDV.abs().pow(2), FDH.abs().pow(2)
        F2D = F2DH + F2DV + 1e-8
        gam = 1e-2 * alpha * alpha / 0.0005
        U1, U2 = X.clone(), X.clone()
        D1, D2 = torch.zeros_like(X), torch.zeros_like(X)

        for _ in range(maxiter):
            V1, V2 = U1 - D1, U2 - D2
            FV1, FV2 = alpha * FDHC * fft2(V1), alpha * FDVC * fft2(V2)
            FR = FCKFKy + FV1 + FV2
            _FKFR_, _F2K_ = (
                splits_and_mean((FK * FR) / F2D, sf),
                splits_and_mean(F2K / F2D, sf),
            )
            _FKFR_FMdiv_FK2_FM = _FKFR_ / (_F2K_ + alpha)
            FCK_FKFR_FMdiv_FK2_FM = FCK * _FKFR_FMdiv_FK2_FM.repeat(1, 1, sf, sf)
            FX = (FR - FCK_FKFR_FMdiv_FK2_FM) / (alpha * F2D)
            X = ifft2(FX).real

            DhX, DvX = ifft2(FX * FDH).real, ifft2(FX * FDV).real
            NU1, NU2 = DhX + D1, DvX + D2
            NU = torch.sqrt(NU1**2 + NU2**2 + 1e-8)

            A = torch.maximum(torch.zeros_like(NU), NU - gam) + 1e-8
            A = A / (A + gam)
            U1, U2 = A * NU1, A * NU2
            D1, D2 = D1 + (DhX - U1), D2 + (DvX - U2)

        return X


class DataNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, alpha, sf, FK, FCK, F2K, FCKFKy):
        FR = FCKFKy + fft2(alpha * X)
        _FKFR_, _F2K_ = (
            splits_and_mean(FK * FR, sf),
            splits_and_mean(F2K, sf),
        )
        _FKFR_FMdiv_FK2_FM = _FKFR_ / (_F2K_ + alpha)
        FCK_FKFR_FMdiv_FK2_FM = FCK * _FKFR_FMdiv_FK2_FM.repeat(1, 1, sf, sf)
        FX = (FR - FCK_FKFR_FMdiv_FK2_FM) / alpha
        return ifft2(FX).real


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
        self.iter_num = 8
        self.v = TVNet()
        self.d = DataNet()
        self.p = ResUNet()
        self.h = HyperNet(in_nc=2, channel=64, out_nc=2 * self.iter_num)

    def forward(self, FK, SKX_n, sigma, sf):
        FCK, F2K = FK.conj(), FK.abs().pow(2)
        Ky = torch.zeros([*FK.shape], device=SKX_n.device, dtype=SKX_n.dtype).repeat(
            1, SKX_n.size(1), 1, 1
        )
        Ky[..., ::sf, ::sf] = SKX_n
        FCKFKy = FCK * fft2(Ky)

        X = self.v(
            SKX_n,
            FK,
            FCK,
            F2K,
            FCKFKy,
            sf,
            0.0005,
        )

        ab = self.h(torch.cat((sigma, torch.full_like(sigma, sf)), 1))

        for i in range(self.iter_num):
            X = self.d(
                X,
                ab[:, i : i + 1],
                sf,
                FK,
                FCK,
                F2K,
                FCKFKy,
            ).float()
            X = X + self.p(
                torch.cat(
                    (X, ab[:, self.iter_num + i].unsqueeze(1).expand_as(X[:, :1])),
                    1,
                )
            )
        return X
