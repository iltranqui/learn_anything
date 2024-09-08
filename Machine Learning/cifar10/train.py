import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import models
from torchmetrics import Accuracy

# LightningModule that can use any model passed to it
class ImageClassificationModel(pl.LightningModule):
    def __init__(self, model, num_classes=10, learning_rate=0.001):
        super(ImageClassificationModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # Replace the final layer for the correct number of classes
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Sequential):
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        acc = self.accuracy(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        acc = self.accuracy(outputs, targets)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


# Example of how to use the model and DataModule in the training loop
if __name__ == "__main__":
    # Choose a model architecture, e.g., VGG16 pretrained
    vgg16_model = models.vgg16(pretrained=True)

    # Instantiate the LightningModule with the chosen model
    model = ImageClassificationModel(model=vgg16_model, num_classes=10, learning_rate=0.001)

    # Instantiate the CIFAR10DataModule or any other DataModule
    data_module = CIFAR10DataModule(batch_size=64)

    # Instantiate the trainer
    trainer = pl.Trainer(max_epochs=10, gpus=1, precision=16)

    # Train the model
    trainer.fit(model, data_module)