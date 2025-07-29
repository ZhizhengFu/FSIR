from typing import Literal
from src.config import Config
from .dataset import FSIRDataset
from torch.utils.data import DataLoader


def get_dataloader(opt: Config, mode: Literal["train", "val", "test"]) -> DataLoader:
    dataset = FSIRDataset(getattr(opt, mode), mode)
    return DataLoader(
        dataset,
        batch_size=opt.batch_size if mode == "train" else 1,
        pin_memory=opt.pin_memory,
        num_workers=opt.num_workers,
        collate_fn=dataset.collate_fn,
        shuffle=(mode == "train"),
    )
