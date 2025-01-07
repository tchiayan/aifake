from torchvision.models import efficientnet_b3 , EfficientNet_B3_Weights
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torch.optim as optim
from torchmetrics import Accuracy , Precision , Recall , AUROC , F1Score

class FakeModelDetection(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(in_features=1536 , out_features=1)

        self.acc_train = Accuracy(task="binary")
        self.acc_val = Accuracy(task="binary")
        self.acc_test = Accuracy(task="binary")

        self.precision_train = Precision(task="binary" , average="macro")
        self.precision_val = Precision(task="binary" , average="macro")
        self.precision_test = Precision(task="binary" , average="macro")

        self.recall_train = Recall(task="binary" , average="macro")
        self.recall_val = Recall(task="binary" , average="macro")
        self.recall_test = Recall(task="binary" , average="macro")

        self.auroc_train = AUROC(task="binary")
        self.auroc_val = AUROC(task="binary")
        self.auroc_test = AUROC(task="binary")

        self.f1_train = F1Score(task="binary" , average="macro")
        self.f1_val = F1Score(task="binary" , average="macro")
        self.f1_test = F1Score(task="binary" , average="macro")

    def forward(self , x):
        return self.model(x)

    def training_step(self, batch , batch_idx):
        images , labels = batch

        output = self(images)
        loss = F.binary_cross_entropy_with_logits(output , labels)
        self.acc_train(output , labels)
        self.precision_train(output , labels)
        self.recall_train(output , labels)
        self.auroc_train(output , labels)
        self.f1_train(output , labels)

        self.log("train_loss" , loss , on_step=False , on_epoch=True , prog_bar=True)
        #self.log("train_acc" , acc , on_step=False , on_epoch=True , prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc" , self.acc_train , prog_bar=True)
        self.log("train_precision" , self.precision_train)
        self.log("train_recall" , self.recall_train)
        self.log("train_auroc" , self.auroc_train)
        self.log("train_f1" , self.f1_train)

    def validation_step(self, batch , batch_idx):
        images , labels = batch

        output = self(images)
        loss = F.binary_cross_entropy_with_logits(output , labels)
        self.acc_val(output , labels)
        self.precision_val(output , labels)
        self.recall_val(output , labels)
        self.auroc_val(output , labels)
        self.f1_val(output , labels)

        self.log("val_loss" , loss , on_step=False , on_epoch=True , prog_bar=True)


    def on_validation_epoch_end(self):
        self.log("val_acc" , self.acc_val , prog_bar=True)
        self.log("val_precision" , self.precision_val )
        self.log("val_recall" , self.recall_val )
        self.log("val_auroc" , self.auroc_val )
        self.log("val_f1" , self.f1_val )


    def test_step(self, batch , batch_idx):
        images , labels = batch
        output = self(images)

        loss = F.binary_cross_entropy_with_logits(output , labels)
        self.acc_test(output , labels)
        self.precision_test(output , labels)
        self.recall_test(output , labels)
        self.auroc_test(output , labels)
        self.f1_test(output , labels)

        self.log("test_loss" , loss)

    def on_test_epoch_end(self):
        self.log("test_acc" , self.acc_test)
        self.log("test_precision" , self.precision_test)
        self.log("test_recall" , self.recall_test)
        self.log("test_auroc" , self.auroc_test)
        self.log("test_f1" , self.f1_test)

    def predict_step(self , batch , batch_idx):
        images , _ = batch
        return self(images)

    def configure_optimizers(self):
        return optim.Adam(self.parameters() , lr=1e-3)
