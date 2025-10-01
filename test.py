import torch
import lightning
from lightning.pytorch.loggers import CSVLogger
from src.engin import FSIR
from src.config import Config
from src.utils import init_seed
from src.data import get_dataloader

checkpoint_path = "weights/example.ckpt"
TYPE = "bicubic"
SIGMA = [0.00]
SF = [2, 3, 4]
CONFIG_NAME = "config"

k_path = (
    "src/kernels/kernels_bicubicx234.pt"
    if TYPE == "bicubic"
    else "src/kernels/kernels_12.pt"
)
K = torch.load(k_path)
config = Config.from_toml(f"{CONFIG_NAME}.toml")
init_seed(config.seed, config.deterministic)
model = FSIR.load_from_checkpoint(checkpoint_path, config=config.trainer)
datasets_config = config.datasets
trainer = lightning.Trainer(
    logger=CSVLogger("logs_test", name=f"test_{TYPE}")
)

for sigma in SIGMA:
    for sf_idx, sf in enumerate(SF):
        for k_idx, kernel in enumerate(K):
            if TYPE == "bicubic" and k_idx != sf_idx:
                continue
            print(f"--- sigma:{sigma} sf:{sf} k_idx:{k_idx} ---")
            kernel = K[k_idx]
            datasets_config.test.sigma = sigma
            datasets_config.test.sf = sf
            datasets_config.test.kernel = kernel
            datasets_config.test.k_idx = k_idx
            test_loader = get_dataloader(datasets_config, "test")
            trainer.test(model, test_loader)
