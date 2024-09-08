import unittest
import torch

# Import your models here
# from your_module import VGG16, GoogLeNet, resnet18

class TestModels(unittest.TestCase):
    def setUp(self):
        # Set up common properties
        self.batch_size = 2
        self.num_classes = 10
        self.input_size = (3, 224, 224)  # Standard size for ImageNet models
        self.input_tensor = torch.randn(self.batch_size, *self.input_size)

    def test_vgg16(self):
        model = VGG16(num_classes=self.num_classes)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_googlenet(self):
        model = GoogLeNet(num_classes=self.num_classes)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_resnet18(self):
        model = resnet18(num_classes=self.num_classes)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_resnet34(self):
        model = resnet34(num_classes=self.num_classes)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_regnet_x_200mf(self):
        model = regnet_x_200mf(num_classes=self.num_classes)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

import unittest
import torch
from torch.utils.data import DataLoader

# Import your DataModule here
# from your_module import CIFAR10DataModule

class TestCIFAR10DataModule(unittest.TestCase):
    def setUp(self):
        # Set up common properties for the DataModule
        self.batch_size = 32
        self.data_dir = './data'
        self.num_workers = 2

    def test_setup_and_prepare_data(self):
        # Initialize the DataModule
        data_module = CIFAR10DataModule(data_dir=self.data_dir, batch_size=self.batch_size, num_workers=self.num_workers)
        
        # Check if setup and prepare_data methods work without errors
        data_module.prepare_data()
        data_module.setup()

        # Check if data loaders are created
        self.assertIsInstance(data_module.train_dataloader(), DataLoader)
        self.assertIsInstance(data_module.val_dataloader(), DataLoader)
        self.assertIsInstance(data_module.test_dataloader(), DataLoader)

    def test_dataloader_output_shape(self):
        # Initialize and set up the DataModule
        data_module = CIFAR10DataModule(data_dir=self.data_dir, batch_size=self.batch_size, num_workers=self.num_workers)
        data_module.prepare_data()
        data_module.setup()

        # Get a batch from the train dataloader
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        inputs, targets = batch

        # Check the shapes of the inputs and targets
        self.assertEqual(inputs.shape, (self.batch_size, 3, 32, 32))
        self.assertEqual(targets.shape, (self.batch_size,))

    def test_num_classes(self):
        # Check that CIFAR-10 has 10 classes
        data_module = CIFAR10DataModule(data_dir=self.data_dir, batch_size=self.batch_size, num_workers=self.num_workers)
        data_module.prepare_data()
        data_module.setup()

        # Check if the number of classes is 10
        train_loader = data_module.train_dataloader()
        _, targets = next(iter(train_loader))
        num_classes = len(torch.unique(targets))
        self.assertEqual(num_classes, 10)

if __name__ == "__main__":
    unittest.main()
