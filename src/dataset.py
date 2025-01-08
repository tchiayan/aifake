from utils import get_transforms
from torch.utils.data import Dataset , DataLoader
import lightning as pl
from pathlib import Path
import os
import pandas as pd
from PIL import Image
import torch

class FakeDataset(Dataset):
    def __init__(self , data_folder:Path , subset: str ,  is_train = False):
        super().__init__()

        self.data_folder = data_folder
        self.is_train = is_train
        self.subset = subset
        self.transforms = get_transforms(train=is_train) # C , H , W

        annoation_file = os.path.join(self.data_folder , f"{subset}.csv")

        if not os.path.exists(annoation_file):
            raise FileNotFoundError(f"{annoation_file} not found")

        self.df = pd.read_csv(annoation_file)



    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_folder  , self.subset , self.df.loc[index , 'image'])
        image = Image.open(image_path)
        image = self.transforms(image)

        label = self.df.loc[index , 'label']
        label = torch.tensor([1] , dtype=torch.float) if label == 'real' else torch.tensor([0] , dtype=torch.float)

        return image , label

class FakeDataModule(pl.LightningDataModule):
    def __init__(self , batch_size:int , data_folder: Path):
        super().__init__()

        self.batch_size = batch_size
        self.data_folder = data_folder

    def setup(self , stage=None):
        self.train_dataset = FakeDataset(data_folder=self.data_folder , subset='train' , is_train=True)
        self.val_dataset = FakeDataset(data_folder=self.data_folder , subset='val' , is_train=False)
        self.test_dataset = FakeDataset(data_folder=self.data_folder , subset='test' , is_train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset , batch_size=self.batch_size , shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset , batch_size=self.batch_size , shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset , batch_size=self.batch_size , shuffle=False)
