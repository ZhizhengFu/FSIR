import lightning
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from src.engin import FSIR
from src.config import Config
from src.data import get_dataloader
from src.utils import init_seed, save_code_snapshot, get_cur_time

CONFIG_NAME = "config"
ckpt_path = ""
config = Config.from_toml(f"{CONFIG_NAME}.toml")
init_seed(config.seed, config.deterministic)
if config.save_snapshots:
    save_code_snapshot(Path("code_snapshots") / get_cur_time(), CONFIG_NAME)

train_loader = get_dataloader(config.datasets, "train")
val_loader = get_dataloader(config.datasets, "val")

checkpoint_callback = ModelCheckpoint(
    monitor="val_psnr",
    mode="max",
    save_top_k=5,
    filename="model-{epoch:02d}-{val_psnr:.2f}",
)
trainer = lightning.Trainer(
    logger=CSVLogger("logs", name="FSIR"),
    # logger=TensorBoardLogger("logs", name="FSIR"),
    callbacks=[checkpoint_callback],
    max_epochs=-1,
    max_steps=-1,
    log_every_n_steps=config.trainer.log_every_n_steps,
    gradient_clip_val=config.trainer.gradient_clip_val,
    gradient_clip_algorithm=config.trainer.gradient_clip_algorithm,
)
model = FSIR(config.trainer)
ckpt_path = Path(ckpt_path) if ckpt_path else None
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
