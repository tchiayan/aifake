from dataset import FakeDataset
from model import FakeModelDetection
from pathlib import Path
from torch.utils.data import DataLoader
import lightning as pl
from sklearn.metrics import classification_report
import torch.nn as nn   
import torch
import os
import mlflow

def main(output_folder: Path):
    
    os.makedirs(output_folder , exist_ok=True)
    
    train_dataset = FakeDataset(data_folder=Path("./data/preprocessed") , subset="train" , is_train=True)
    train_loader = DataLoader(train_dataset , batch_size=32 , shuffle=True)

    val_dataset = FakeDataset(data_folder=Path("./data/preprocessed") , subset="val" , is_train=False)
    val_loader = DataLoader(val_dataset , batch_size=32 , shuffle=False)
    
    test_dataset = FakeDataset(data_folder=Path("./data/preprocessed") , subset="test" , is_train=False)
    test_loader = DataLoader(test_dataset , batch_size=32 , shuffle=False)
    
    model = FakeModelDetection()
    mlflow.set_experiment("EfficientNet_B3")
    
    with mlflow.start_run() as run:
        trainer = pl.Trainer(
            max_epochs=5 , 
            logger=[], 
            enable_checkpointing=False , 
        )
        
        trainer.fit(model , train_loader , val_loader)
        trainer.test(model , test_loader)
        
        # Generate sklearn classification report 
        y_true = []
        for _ , labels in test_loader:
            y_true.append(labels) 
        y_true = torch.concat(y_true , dim=0)
        print(y_true.shape)
        y_pred = trainer.predict(model , test_loader)
        y_pred = torch.concat(y_pred , dim=0)
        y_pred = (nn.Sigmoid()(y_pred) > 0.5).int()
        print(y_pred.shape)
        
        
        report = classification_report(y_true , y_pred)
        with open(output_folder / "classification_report.txt" , "w") as f:
            f.write(report)
            print(report)
            
        # log the classification report to mlflow
        mlflow.log_artifact(output_folder / "classification_report.txt")
    
if __name__ == "__main__":
    main()