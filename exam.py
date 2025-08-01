import torch
import torch.nn.functional as F
from torch.fft import fft2, ifft2
from torchvision.io import read_image

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from src.utils import KernelSynthesizer, imshow
from src.model import TVNet, DataNet

# # K_TYPE = "gaussian"
# K_TYPE = "motion"
# sf = 4
# SIGMA = 0.00
# torch.manual_seed(40)
# X = read_image("datasets/test/Set5/butterfly.png").unsqueeze(0) / 255

# psnr = PeakSignalNoiseRatio(data_range=(0, 1))
# ssim = StructuralSimilarityIndexMeasure()
# K = getattr(KernelSynthesizer(), f"gen_{K_TYPE}_kernel")().unsqueeze(0).unsqueeze(0)
# sigma = torch.full((1, 1, 1, 1), SIGMA)
# H, W = X.shape[-2:]
# H_r, W_r = H % sf, W % sf
# X = X[..., : H - H_r, : W - W_r]
# psf_h, psf_w = K.shape[-2:]
# target_h, target_w = X.shape[-2:]
# padding = (0, target_w - psf_w, 0, target_h - psf_h)
# otf = F.pad(K, padding).double()
# shifts = (-(psf_h // 2), -(psf_w // 2))
# FK = fft2(otf.roll(shifts, (-2, -1)))
# KX = ifft2(fft2(X) * FK).real
# SKX = KX[..., ::sf, ::sf]
# SKX_n = SKX + torch.randn_like(SKX) * sigma
# v = TVNet()
# FCK, F2K = FK.conj(), FK.abs().pow(2)
# Ky = torch.zeros([*FK.shape], device=SKX_n.device, dtype=SKX_n.dtype).repeat(
#     1, SKX_n.size(1), 1, 1
# )
# Ky[..., ::sf, ::sf] = SKX_n
# FCKFKy = FCK * fft2(Ky)
# out = v(
#     SKX_n,
#     FK,
#     FCK,
#     F2K,
#     FCKFKy,
#     sf,
#     0.0005,
# )

# print("PSNR: ", psnr(X, out))
# print("SSIM: ", ssim(X, out))
# imshow([X, out, K])


import matplotlib.pyplot as plt

def analyze_and_visualize_frequency_psnr(original, reconstructed, data_range=(0, 1)):
    print("\n--- 开始进行频域分析 ---")

    original = original.detach().cpu().squeeze()
    reconstructed = reconstructed.detach().cpu().squeeze()

    C, H, W = original.shape

    F_orig = fft2(original)
    F_recon = fft2(reconstructed)

    freq_x = torch.fft.fftfreq(W)
    freq_y = torch.fft.fftfreq(H)
    grid_x, grid_y = torch.meshgrid(freq_x, freq_y, indexing="xy")

    radius = torch.sqrt(grid_x**2 + grid_y**2) / 0.5

    bands = {
        "low1 (0-5%)": (0.0, 0.05),
        "low2 (5%-10%)": (0.05, 0.1),
        "mid (10-20%)": (0.1, 0.2),
        "high (20-50%)": (0.2, 0.5),
    }

    psnr_scores = {}

    fig_imgs, axs_imgs = plt.subplots(len(bands), 3, figsize=(15, 13))
    fig_imgs.suptitle(
        "visualize (origin vs. reconstuct vs. diff)", fontsize=18, y=0.98
    )

    for i, (band_name, (start_r, end_r)) in enumerate(bands.items()):
        mask = (radius >= start_r) & (radius < end_r)

        F_orig_band = F_orig * mask
        F_recon_band = F_recon * mask

        orig_band_img = ifft2(F_orig_band).real
        recon_band_img = ifft2(F_recon_band).real

        mse = F.mse_loss(orig_band_img, recon_band_img)

        if mse > 1e-10:
            band_psnr = -10 * torch.log10(mse)
        else:
            band_psnr = float("inf")

        psnr_scores[band_name] = band_psnr
        print(f"band '{band_name}'  PSNR: {psnr_scores[band_name]:.2f} dB")

        diff_img = torch.abs(orig_band_img - recon_band_img)

        im1 = axs_imgs[i, 0].imshow(
            orig_band_img.permute(1, 2, 0).cpu().numpy(), cmap="gray"
        )
        axs_imgs[i, 0].set_title(f"{band_name} - origin", fontsize=12)
        fig_imgs.colorbar(im1, ax=axs_imgs[i, 0], fraction=0.046, pad=0.04)

        im2 = axs_imgs[i, 1].imshow(
            recon_band_img.permute(1, 2, 0).cpu().numpy(), cmap="gray"
        )
        axs_imgs[i, 1].set_title(f"{band_name} - reconstruct", fontsize=12)
        fig_imgs.colorbar(im2, ax=axs_imgs[i, 1], fraction=0.046, pad=0.04)

        im3 = axs_imgs[i, 2].imshow(diff_img.permute(1, 2, 0).cpu().numpy(), cmap="hot")
        axs_imgs[i, 2].set_title(f"{band_name} - diff", fontsize=12)
        fig_imgs.colorbar(im3, ax=axs_imgs[i, 2], fraction=0.046, pad=0.04)

        for ax in axs_imgs[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig_imgs.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore

    return psnr_scores


if __name__ == "__main__":

    K_TYPE = "motion"
    sf = 4
    SIGMA = 0.00
    torch.manual_seed(40)

    X = read_image("datasets/test/Set5/butterfly.png").unsqueeze(0) / 255


    psnr = PeakSignalNoiseRatio(
        data_range=(0, 1), reduction="elementwise_mean"
    ).double()
    ssim = StructuralSimilarityIndexMeasure(data_range=(0, 1)).double()


    K = (
        getattr(KernelSynthesizer(), f"gen_{K_TYPE}_kernel")()
        .unsqueeze(0)
        .unsqueeze(0)
        .double()
    )
    sigma = torch.full((1, 1, 1, 1), SIGMA).double()


    H, W = X.shape[-2:]
    H_r, W_r = H % sf, W % sf
    X = X[..., : H - H_r, : W - W_r]


    psf_h, psf_w = K.shape[-2:]
    target_h, target_w = X.shape[-2:]
    padding = (0, target_w - psf_w, 0, target_h - psf_h)
    otf = F.pad(K, padding)
    shifts = (-(psf_h // 2), -(psf_w // 2))
    FK = fft2(otf.roll(shifts, (-2, -1)))

    KX = ifft2(fft2(X) * FK).real
    SKX = KX[..., ::sf, ::sf]
    SKX_n = SKX + torch.randn_like(SKX) * sigma

    v = TVNet()
    # v = DataNet()

    FCK, F2K = FK.conj(), FK.abs().pow(2)
    Ky = torch.zeros_like(FK).repeat(1, SKX_n.size(1), 1, 1)
    Ky[..., ::sf, ::sf] = SKX_n
    FCKFKy = FCK * fft2(Ky)

    regularization_lambda = 0.005
    out = v(
        SKX_n,
        FK,
        FCK,
        F2K,
        FCKFKy,
        sf,
        regularization_lambda,
    )
    # out = v(F.interpolate(SKX_n, scale_factor=sf, mode="nearest"), 1e-4, sf, FK, FCK, F2K, FCKFKy)

    overall_psnr = psnr(out, X)
    overall_ssim = ssim(out, X)
    print("--- 整体图像质量评估 ---")
    print(f"整体 PSNR: {overall_psnr.item():.2f} dB")
    print(f"整体 SSIM: {overall_ssim.item():.4f}")

    analyze_and_visualize_frequency_psnr(X, out, data_range=(0, 1))
    analyze_and_visualize_frequency_psnr(KX, out, data_range=(0, 1))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(X.squeeze().permute(1, 2, 0).cpu().numpy(), cmap="gray")
    axs[0].set_title("Ground Truth")
    axs[0].axis("off")

    axs[1].imshow(SKX_n.squeeze().permute(1, 2, 0).cpu().numpy(), cmap="gray")
    axs[1].set_title(f"LOW IMAGE\n(PSF: {K_TYPE}, Noise: {SIGMA})")
    axs[1].axis("off")

    axs[2].imshow(out.squeeze().permute(1, 2, 0).cpu().numpy(), cmap="gray")
    axs[2].set_title(f"RECONSTUCT\n(Overall PSNR: {overall_psnr.item():.2f} dB)")
    axs[2].axis("off")
    plt.show()
