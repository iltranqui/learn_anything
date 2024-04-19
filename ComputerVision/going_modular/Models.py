
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch import Tensor
from typing import Any, Callable, List, Optional, Tuple


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VGG16(nn.Module):
    """
    VGG16 model implementation.

    Args:
        num_classes (int): Number of output classes.

    Attributes:
        layer1 (nn.Sequential): First convolutional layer.
        layer2 (nn.Sequential): Second convolutional layer.
        layer3 (nn.Sequential): Third convolutional layer.
        layer4 (nn.Sequential): Fourth convolutional layer.
        layer5 (nn.Sequential): Fifth convolutional layer.
        layer6 (nn.Sequential): Sixth convolutional layer.
        layer7 (nn.Sequential): Seventh convolutional layer.
        layer8 (nn.Sequential): Eighth convolutional layer.
        layer9 (nn.Sequential): Ninth convolutional layer.
        layer10 (nn.Sequential): Tenth convolutional layer.
        layer11 (nn.Sequential): Eleventh convolutional layer.
        layer12 (nn.Sequential): Twelfth convolutional layer.
        layer13 (nn.Sequential): Thirteenth convolutional layer.
        fully_connected_1 (nn.Sequential): First fully connected layer.
        fully_connected_2 (nn.Sequential): Second fully connected layer.
        fully_connected_3 (nn.Sequential): Third fully connected layer.

    Methods:
        forward(x): Forward pass of the model.

    """
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
    """
    VGG19 model implementation.

    Args:
        num_classes (int): The number of output classes.

    Attributes:
        conv1_1 (nn.Sequential): First convolutional layer block.
        conv1_2 (nn.Sequential): Second convolutional layer block.
        conv2_1 (nn.Sequential): Third convolutional layer block.
        conv2_2 (nn.Sequential): Fourth convolutional layer block.
        conv3_1 (nn.Sequential): Fifth convolutional layer block.
        conv3_2 (nn.Sequential): Sixth convolutional layer block.
        conv3_3 (nn.Sequential): Seventh convolutional layer block.
        conv3_4 (nn.Sequential): Eighth convolutional layer block.
        conv4_1 (nn.Sequential): Ninth convolutional layer block.
        conv4_2 (nn.Sequential): Tenth convolutional layer block.
        conv4_3 (nn.Sequential): Eleventh convolutional layer block.
        conv4_4 (nn.Sequential): Twelfth convolutional layer block.
        conv5_1 (nn.Sequential): Thirteenth convolutional layer block.
        conv5_2 (nn.Sequential): Fourteenth convolutional layer block.
        conv5_3 (nn.Sequential): Fifteenth convolutional layer block.
        conv5_4 (nn.Sequential): Sixteenth convolutional layer block.
        fully_connected_1 (nn.Sequential): First fully connected layer block.
        fully_connected_2 (nn.Sequential): Second fully connected layer block.
        fully_connected_3 (nn.Sequential): Third fully connected layer block.

    Methods:
        forward(x): Performs forward pass through the network.

    """
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
    """
    AlexNet model implementation.
    It works ! -> Training

    Args:
        num_classes (int): Number of output classes. Default is 10.

    Attributes:
        conv1 (nn.Sequential): First convolutional layer.
        conv2 (nn.Sequential): Second convolutional layer.
        conv3 (nn.Sequential): Third convolutional layer.
        conv4 (nn.Sequential): Fourth convolutional layer.
        conv5 (nn.Sequential): Fifth convolutional layer.
        fully_connected (nn.Sequential): Fully connected layers.

    """

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # Convolutional layer 1: input (3, 224, 224), output (96, 55, 55)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, padding=0, stride=4),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Convolutional layer 2: input (96, 27, 27), output (256, 27, 27)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Convolutional layer 3: input (256, 13, 13), output (384, 13, 13)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU()
        )

        # Convolutional layer 4: input (384, 13, 13), output (384, 13, 13)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU()
        )

        # Convolutional layer 5: input (384, 13, 13), output (256, 13, 13)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Fully connected layers: input (256 * 6 * 6), output (num_classes)
        self.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features=256*5*5, out_features=4096),
                    nn.ReLU())
        
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU())
        
        self.fc2= nn.Sequential(
            nn.Linear(in_features=4096, out_features=num_classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    
class SegNet(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super(SegNet, self).__init__()

        #SegNet Architecture
        #Takes input of size in_chn = 3 (RGB images have 3 channels)
        #Outputs size label_chn (N # of classes)

        #ENCODING consists of 5 stages
        #Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
        #Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

        #General Max Pool 2D for ENCODING layers
        #Pooling indices are stored for Upsampling in DECODING layers+

        # BN_momentum  is for the batchNormalization to reduce overfitting and outliers

        self.in_chn = in_chn  # Input: N*N*3
        self.out_chn = out_chn 

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)              # Output: N/2*N/2*3

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)     # O: N/2*N/2*64
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)                   # O: N/2*2/N*64 
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)              # O: N/2*N/2*64
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)                   # O: N/2*N/2*64

        # here there is the MAxPooling layer, it is added later ->               # O: N/4*N/4*64

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)             # O: N/4*N/4*128
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)                  # O: N/4*N/4*128
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)            # O: N/4*N/4*128
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)                  # O: N/4*N/4*128

                                                                                 #  O: N/8*N/*256
                                                                                 #  
        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)            #  O: N/8*N/*256
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)                  #  O: N/8*N/*256
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)            #  O: N/8*N/*256
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)                  #  O: N/8*N/*256
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)            #  O: N/8*N/*256
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)                  #  O: N/8*N/*256

                                                                                 #  MaxPooling: N/16*N/16*256

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)            #  O: N/16*N/*512
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)            #  O: N/16*N/*512
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)            #  O: N/16*N/*512
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

                                                                                 #  MaxPooling: N/32*N/32*512

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)            #  O: N/32*N/32*512
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)            #  O: N/32*N/32*512
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)            #  O: N/32*N/32*512
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)


        #DECODING consists of 5 stages
        #Each stage corresponds to their respective counterparts in ENCODING

        #General Max Pool 2D/Upsampling for DECODING layers
        self.MaxDe = nn.MaxUnpool2d(2, stride=2) 

        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  #  O: N/32*N/32*512
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  #  O: N/32*N/32*512
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  #  O: N/32*N/32*512
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  #  O: N/16*N/*512
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  #  O: N/16*N/*512
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  #  O: N/16*N/*256
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  #  O: N/8*N/*256
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  #  O: N/8*N/*256
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  #  O: N/8*N/*128
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  #  O: N/4*N/4*128
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  #  O: N/4*N/4*64
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  #  O: N/2*N/2*64
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)  #  O: N/2*N/2*self.out_chn
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

    def forward(self, x):
                #ENCODE LAYERS
        #Stage 1
        x = nn.ReLU(self.BNEn11(self.ConvEn11(x))) 
        x = nn.ReLU(self.BNEn12(self.ConvEn12(x))) 
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        #Stage 2
        x = nn.ReLU(self.BNEn21(self.ConvEn21(x))) 
        x = nn.ReLU(self.BNEn22(self.ConvEn22(x))) 
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        #Stage 3
        x = nn.ReLU(self.BNEn31(self.ConvEn31(x))) 
        x = nn.ReLU(self.BNEn32(self.ConvEn32(x))) 
        x = nn.ReLU(self.BNEn33(self.ConvEn33(x)))   
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        #Stage 4
        x = nn.ReLU(self.BNEn41(self.ConvEn41(x))) 
        x = nn.ReLU(self.BNEn42(self.ConvEn42(x))) 
        x = nn.ReLU(self.BNEn43(self.ConvEn43(x)))   
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        #Stage 5
        x = nn.ReLU(self.BNEn51(self.ConvEn51(x))) 
        x = nn.ReLU(self.BNEn52(self.ConvEn52(x))) 
        x = nn.ReLU(self.BNEn53(self.ConvEn53(x)))   
        x, ind5 = self.MaxEn(x)
        size5 = x.size()

        #DECODE LAYERS
        #Stage 5
        x = self.MaxDe(x, ind5, output_size=size4)
        x = nn.ReLU(self.BNDe53(self.ConvDe53(x)))
        x = nn.ReLU(self.BNDe52(self.ConvDe52(x)))
        x = nn.ReLU(self.BNDe51(self.ConvDe51(x)))

        #Stage 4
        x = self.MaxDe(x, ind4, output_size=size3)
        x = nn.ReLU(self.BNDe43(self.ConvDe43(x)))
        x = nn.ReLU(self.BNDe42(self.ConvDe42(x)))
        x = nn.ReLU(self.BNDe41(self.ConvDe41(x)))

        #Stage 3
        x = self.MaxDe(x, ind3, output_size=size2)
        x = nn.ReLU(self.BNDe33(self.ConvDe33(x)))
        x = nn.ReLU(self.BNDe32(self.ConvDe32(x)))
        x = nn.ReLU(self.BNDe31(self.ConvDe31(x)))

        #Stage 2
        x = self.MaxDe(x, ind2, output_size=size1)
        x = nn.ReLU(self.BNDe22(self.ConvDe22(x)))
        x = nn.ReLU(self.BNDe21(self.ConvDe21(x)))

        #Stage 1
        x = self.MaxDe(x, ind1)
        x = nn.ReLU(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)

        x = nn.Softmax(x, dim=1)

        return x


    # From https://github.com/vinceecws/SegNet_PyTorch
    @staticmethod 
    def save_checkpoint(state, path):
        torch.save(state, path)
        print("Checkpoint saved at {}".format(path))

    @staticmethod
    def Train(trainloader, path=None): #epochs is target epoch, path is provided to load saved checkpoint

        model = SegNet()
        optimizer = optim.SGD(model.parameters(), lr=hyperparam.lr, momentum=hyperparam.momentum)
        loss_fn = nn.CrossEntropyLoss()
        run_epoch = hyperparam.epochs

        if path == None:
            epoch = 0
            path = os.path.join(os.getcwd(), 'segnet_weights.pth.tar')
            print("Creating new checkpoint '{}'".format(path))
        else:
            if os.path.isfile(path):
                print("Loading checkpoint '{}'".format(path))
                checkpoint = torch.load(path)
                epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
            else:
                print("No checkpoint found at '{}'".format(path))
                

        for i in range(1, run_epoch + 1):
            print('Epoch {}:'.format(i))
            sum_loss = 0.0

            for j, data in enumerate(trainloader, 1):
                images, labels = data
                optimizer.zero_grad()
                output = model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()

                print('Loss at {} mini-batch: {}'.format(j, loss.item()/trainloader.batch_size))

            print('Average loss @ epoch: {}'.format((sum_loss/j*trainloader.batch_size)))

        print("Training complete. Saving checkpoint...")
        Train.save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, path)

class GoogleNet(nn.Module):
    """
    GoogleNet model implementation.
    Example COde is here: https://pytorch.org/vision/main/_modules/torchvision/models/googlenet.html
    Args:
        num_classes (int): Number of output classes. Default is 10.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        conv4 (nn.Conv2d): Fourth convolutional layer.
        conv5 (nn.Conv2d): Fifth convolutional layer.
        conv6 (nn.Conv2d): Sixth convolutional layer.
        conv7 (nn.Conv2d): Seventh convolutional layer.
        conv8 (nn.Conv2d): Eighth convolutional layer.
        conv9 (nn.Conv2d): Ninth convolutional layer.
        conv10 (nn.Conv2d): Tenth convolutional layer.
        conv11 (nn.Conv2d): Eleventh convolutional layer.
        conv12 (nn.Conv2d): Twelfth convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        maxpool (nn.MaxPool2d): Max pooling layer.
        avgpool (nn.AvgPool2d): Average pooling layer.
        dropout (nn.Dropout): Dropout layer.
        relu (nn.ReLU): ReLU activation function.
        softmax (nn.Softmax): Softmax activation function.
    """

    def __init__(self, num_classes: int =10,  aux_logits: bool = True, transform_input: bool = False, dropout: float = 0.2, dropout_aux: float = 0.7):
        super(GoogleNet, self).__init__()

        #self.transform_input = transform_input

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, pool_proj=32)
        self.inception3b = Inception(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192, ch5x5red=32, ch5x5=96, pool_proj=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(in_channels=512, num_classes=num_classes, dropout=dropout_aux)
            self.aux2 = InceptionAux(in_channels=528, num_classes=num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_aux)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        #return torch.cat([x, aux2, aux1], dim=1)
        return x, aux2, aux1
    
    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

class Inception(nn.Module):
    """
    An Inception Module is an image model block that aims to approximate an optimal local sparse structure in a CNN.
    Put simply, it allows for us to use multiple types of filter size, instead of being restricted to a single filter size, in a single image block, which we then concatenate and pass onto the next layer.

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        in_channels: int,  # channels in input
        ch1x1: int, # the number of filers for conv1x1
        ch3x3red: int, # the number of filters for conv3x3
        ch3x3: int, # the number of filters for 
        ch5x5red: int, # the number of filters for conv5x5
        ch5x5: int, # the number of filters for conv5x5
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        
        super().__init__()
        if conv_block is None:  # making sure that the BasicOnc 2 block is defined
            conv_block = BasicConv2d

        # Convolution Block with 1x1 Kernel Size 
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)


        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),  # 1st passing a convolution layer with 1 size kernel
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)  #2nd passing the 2nd convolution layer with 3 size kernel
        )

        self.branch3 = nn.Sequential( 
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.  -> so let's just leave it let's say
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class InceptionAux(nn.Module):
    """
    An auxiliary classifier is a classification block that stems from one of the intermediate layers.
    We take its predictions into account to help the network to propagate gradients through the more recent layers.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, 
        in_channels: int, # input channels
        num_classes: int, # Num of classes to classify 
        conv_block: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.7,
    ) -> None:
        
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv1x1 = conv_block(in_channels=in_channels, out_channels=128, kernel_size=1)

        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv1x1(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)
        # return torch.cat([x], dim=1)
        return x


class BasicConv2d(nn.Module):
    """"
    OK, correct ! 
    Convolution Block followed by a Normalization Layer and then simply followed by a  Relu Activation function
    
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class ResNet(nn.Module):
    """
    ResNet model implementation.
    Still to implement

    Args:
        num_classes (int): Number of output classes. Default is 10.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        conv4 (nn.Conv2d): Fourth convolutional layer.
        conv5 (nn.Conv2d): Fifth convolutional layer.
        conv6 (nn.Conv2d): Sixth convolutional layer.
        conv7 (nn.Conv2d): Seventh convolutional layer.
        conv8 (nn.Conv2d): Eighth convolutional layer.
        conv9 (nn.Conv2d): Ninth convolutional layer.
        conv10 (nn.Conv2d): Tenth convolutional layer.
        conv11 (nn.Conv2d): Eleventh convolutional layer.
        conv12 (nn.Conv2d): Twelfth convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        maxpool (nn.MaxPool2d): Max pooling layer.
        avgpool (nn.AvgPool2d): Average pooling layer.
        dropout (nn.Dropout): Dropout layer.
        relu (nn.ReLU): ReLU activation function.
        softmax (nn.Softmax): Softmax activation function.
    """

    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=480, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=480, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.maxpool(x)
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.maxpool(x)
        x = self.relu(self.conv12(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    
class ChannelBoastedCNN(nn.Module):
    """
    Channel Boasted CNN model implementation.

    Args:
        num_classes (int): Number of output classes. Default is 10.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        conv4 (nn.Conv2d): Fourth convolutional layer.
        conv5 (nn.Conv2d): Fifth convolutional layer.
        conv6 (nn.Conv2d): Sixth convolutional layer.
        conv7 (nn.Conv2d): Seventh convolutional layer.
        conv8 (nn.Conv2d): Eighth convolutional layer.
        conv9 (nn.Conv2d): Ninth convolutional layer.
        conv10 (nn.Conv2d): Tenth convolutional layer.
        conv11 (nn.Conv2d): Eleventh convolutional layer.
        conv12 (nn.Conv2d): Twelfth convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        maxpool (nn.MaxPool2d): Max pooling layer.
        avgpool (nn.AvgPool2d): Average pooling layer.
        dropout (nn.Dropout): Dropout layer.
        relu (nn.ReLU): ReLU activation function.
        softmax (nn.Softmax): Softmax activation function.
    """

    def __init__(self, num_classes=10):
        super(ChannelBoastedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=480, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=480, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

