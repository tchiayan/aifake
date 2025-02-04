import pytest
from src.dataset import FakeDataset , FakeDataModule
from torch.utils.data import DataLoader
import torch
import os
from pathlib import Path

def test_datsaset():
    print(os.getcwd())
    dataset = FakeDataset(data_folder="./data/preprocessed", subset="train", is_train=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        images , label = batch
        assert len(images.shape) == 4
        assert images.shape[0] == 32
        assert images.shape[1] == 3
        assert images.shape[2] == 256
        assert images.shape[3] == 256
        assert isinstance(images , torch.Tensor)

        assert len(label.shape) == 2
        assert label.shape[0] == 32
        assert label.shape[1] == 1
        assert isinstance(label , torch.Tensor)

        break

def test_data_module():
    data_module = FakeDataModule(batch_size=32 , data_folder=Path("./data/preprocessed"))
    data_module.setup()
    train_loader = data_module.train_dataloader()

    for batch in train_loader:
        images , label = batch
        assert len(images.shape) == 4
        assert images.shape[0] == 32
        assert images.shape[1] == 3
        assert images.shape[2] == 256
        assert images.shape[3] == 256
        assert isinstance(images , torch.Tensor)

        assert len(label.shape) == 2
        assert label.shape[0] == 32
        assert label.shape[1] == 1
        assert isinstance(label , torch.Tensor)

        break
