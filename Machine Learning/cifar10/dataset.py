import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        # Download the CIFAR-10 dataset
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Split the CIFAR-10 dataset into train, val, and test sets
        if stage == 'fit' or stage is None:
            full_train_dataset = CIFAR10(root=self.data_dir, train=True, transform=self.transform)
            val_size = 5000
            train_size = len(full_train_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.test_dataset = CIFAR10(root=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

if __name__ == '__main__':
    data_module = CIFAR10DataModule()
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    for batch in train_loader:
        inputs, targets = batch
        print(inputs.shape, targets.shape)
        break

    for batch in val_loader:
        inputs, targets = batch
        print(inputs.shape, targets.shape)
        break

    for batch in test_loader:
        inputs, targets = batch
        print(inputs.shape, targets.shape)
        break
    # Example usage
    data_module = CIFAR10DataModule(batch_size=64)
    model = CIFAR10Model(num_classes=10)

    trainer = pl.Trainer(max_epochs=10, gpus=1, precision=16)
    trainer.fit(model, data_module)
