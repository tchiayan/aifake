from utils import get_transforms 
from torch.utils.data import Dataset 
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
        
        annoation_file = self.data_folder / f"{subset}.csv"
        
        if not os.path.exists(annoation_file):
            raise FileNotFoundError(f"{annoation_file} not found")
        
        self.df = pd.read_csv(annoation_file)
    

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_path = self.data_folder / self.subset / self.df.loc[index , 'image']
        image = Image.open(image_path)
        image = self.transforms(image)
        
        label = self.df.loc[index , 'label']
        label = torch.tensor([1] , dtype=torch.float) if label == 'real' else torch.tensor([0] , dtype=torch.float)
        
        return image , label