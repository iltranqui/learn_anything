
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
            nn.MaxPool2d(kernel_size=3, stride=2)                                               # Output: 13*13*256
        )

        self.conv3 = nn.Sequetial(
            nn.Conv2d(in_channels=265, out_channels=384, kernel_size=3, stride=1, padding=1),   # Input: 13*13*256
            nn.BatchNorm2d(num_features=256),                                                   # Output: 13*13*384
            nn.ReLU()
        )

        self.conv4 = nn.Sequetial(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),   # Input: 13*13*384
            nn.BatchNorm2d(num_features=256),                                                   # Output: 13*13*384
            nn.ReLU()
        )

        self.conv5 = nn.Sequetial(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),   # Input: 13*13*384
            nn.BatchNorm2d(num_features=256),                                                   # Input: 13*13*256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)                                               # Output: 6*6*256        
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

        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
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


