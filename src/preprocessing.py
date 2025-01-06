from sklearn.model_selection import train_test_split 
from pathlib import Path
import pandas as pd
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import shutil
import logging

def main(
    data_folder: Path ,
    output_folder: Path , 
    annotation_filepath: Path , 
    train_ratio: float, 
    val_ratio: float , 
    test_ratio: float):
    """
    Split the data into train, validation and test sets 
    """
    
    # Load the data 
    df = pd.read_csv(annotation_filepath)
    
    # create columns "mean_rgb" 
    df['mean_rgb'] = 0.0
    
    for idx , row in df.iterrows(): 
        filepath = os.path.join(data_folder , 'Train' , row['image'])
        shutil.copy(filepath , os.path.join(output_folder , 'train' , row['image']))
        image = Image.open(filepath)
        image_np = np.array(image)
        mean_rgb = image_np.mean()
        df.loc[idx , 'mean_rgb'] = mean_rgb
        
    # Split the data 
    train_df  , val_df = train_test_split(df , train_size=train_ratio) 
    val_df , test_df = train_test_split(val_df , train_size=(val_ratio/(test_ratio+val_ratio)))
    
    # Copy the images to the corresponding folders
    os.makedirs(os.path.join(output_folder , 'train') , exist_ok=True)
    os.makedirs(os.path.join(output_folder , 'val') , exist_ok=True)
    os.makedirs(os.path.join(output_folder , 'test') , exist_ok=True)
    for _df , folder in [(train_df , 'train') , (val_df , 'val') , (test_df , 'test')]: 
        for idx , row in _df.iterrows(): 
            filepath = os.path.join(data_folder , 'Train' , row['image'])
            shutil.copy(filepath , os.path.join(output_folder , folder , row['image']))

    # Save the data
    os.makedirs(output_folder , exist_ok=True)
    train_df.to_csv(output_folder / 'train.csv' , index=False)  
    val_df.to_csv(output_folder / 'val.csv' , index=False)
    test_df.to_csv(output_folder / 'test.csv' , index=False)
    
    # Plot the mean_rgb distribution for train , val and test sets 
    fig , ax = plt.subplots( 1, 1 , figsize=(20, 20))
    ax.hist(train_df['mean_rgb']  , alpha=0.5 , label='train' , density=True)
    ax.hist(val_df['mean_rgb']  , alpha=0.5 , label='val' , density=True)
    ax.hist(test_df['mean_rgb'] , alpha=0.5 , label='test' , density=True)
    ax.legend()
    
    plt.savefig(output_folder / 'mean_rgb_distribution.png')
    
    # Plot the mean_rgb distribution comparison between actual and fake images
    fig, ax = plt.subplots(3 , 1 , figsize=(20, 20))
    for row_idx , (title , _df) in enumerate([ ('train' , train_df) , ('val' , val_df) , ('test' , test_df)]): 
        ax[row_idx].hist(_df[_df['label'] == 'editada']['mean_rgb']  , alpha=0.5 , label='editada' , density=True)
        ax[row_idx].hist(_df[_df['label'] == 'real']['mean_rgb']  , alpha=0.5 , label='original' , density=True)
        ax[row_idx].legend()
        ax[row_idx].set_title(f'{title} mean_rgb distribution comparison')
        
    plt.savefig(output_folder / 'mean_rgb_distribution_comparison.png')
        
        
if __name__ == '__main__':
    main(
        data_folder=Path('data/raw') , 
        output_folder=Path('data/preprocessed') , 
        annotation_filepath=Path('data/raw/train.csv') , 
        train_ratio=0.8 , 
        val_ratio=0.1 , 
        test_ratio=0.1
    )