
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VGG16(nn.Module):
    def __init__(self, num_classes):   # Init -> necessary for the class
        super(VGG16, self).__init__()    # superimposing voer the vgg17 of torch
        self.layer1 = nn.Sequential(        
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),   # 3 input channles equivalento to the 3 color channels
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(        
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(        
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(        
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(        
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(        
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(        
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer8 = nn.Sequential(        
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(        
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.layer10 = nn.Sequential(        
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer11 = nn.Sequential(        
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.layer12 = nn.Sequential(        
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer13 = nn.Sequential(        
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.fully_connected_1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=7*7*512,out_features=4096),
            nn.ReLU()
        )
        self.fully_connected_2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU()
        )
        self.fully_connected_3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=num_classes)
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fully_connected_1(out)
        out = self.fully_connected_2(out)
        out = self.fully_connected_3(out)
        return out

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )
        self.conv3_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )
        self.conv4_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.conv5_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )
        self.conv5_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )
        self.conv5_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fully_connected_1 = nn.Sequential(
            nn.Linear(in_features=7*7*512, out_features=4096)  
            # Not in_features=4096 -> multiplication matrix wrong
        )
        self.fully_connected_2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096)
        )
        self.fully_connected_3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.conv1_2(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.conv3_3(out)
        out = self.conv3_4(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.conv4_3(out)
        out = self.conv4_4(out)

        out = self.conv5_1(out)
        out = self.conv5_2(out)
        out = self.conv5_3(out)
        out = self.conv5_4(out)

        out = out.reshape(out.size(0), -1)
        out = self.fully_connected_1(out)
        out = self.fully_connected_2(out)
        out = self.fully_connected_3(out)
        return out

class AlexNet(nn.Module):
    # Input 227*227*3
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, padding=0, stride=4),     # # Input 227*227*3
            nn.BatchNorm2d(num_features=96),                                                    # Output: 55*55*96    
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)                                               # Output: 27*27*96
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=1),    # Input: 27*27*96
            nn.BatchNorm2d(num_features=256),                                                   # Output: 27*27*256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)                                               tput: 
        )

        self.conv3 = nn.Sequetial(
            nn.Conv2d(in_channels=265, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        self.conv4 = nn.Sequetial(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        self.conv5 = nn.Sequetial(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fully_connected = nn.Sequetial(
            nn.Droput(0.5),
            nn.Linear(in_features=9216,out_features=4096),
            nn.ReLU(),
            nn.Droput(0.5),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(),
            nn.Droput(0.5),
            nn.Linear(in_features=1000,out_features=num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fully_connected(out)
        return out