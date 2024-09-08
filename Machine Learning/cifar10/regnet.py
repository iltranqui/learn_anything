import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleStem(nn.Module):
    """Simple stem layer consisting of a single convolutional layer followed by batch normalization and ReLU."""
    def __init__(self, in_channels, out_channels):
        super(SimpleStem, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BottleneckBlock(nn.Module):
    """Bottleneck block with 1x1, 3x3, and 1x1 convolutions."""
    def __init__(self, in_channels, out_channels, stride=1, group_width=1):
        super(BottleneckBlock, self).__init__()
        bottleneck_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, groups=group_width, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class RegNet(nn.Module):
    """RegNet model consisting of a stem, a series of blocks, and a head."""
    def __init__(self, stem_channels, block_channels, num_blocks, num_classes=1000, group_width=1):
        super(RegNet, self).__init__()
        self.stem = SimpleStem(3, stem_channels)

        layers = []
        in_channels = stem_channels
        for i in range(len(block_channels)):
            layers.append(self._make_layer(BottleneckBlock, in_channels, block_channels[i], num_blocks[i], stride=2 if i > 0 else 1, group_width=group_width))
            in_channels = block_channels[i]
        self.blocks = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block_channels[-1], num_classes)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1, group_width=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride, group_width))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, group_width=group_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Example of creating a RegNet variant
def regnet_x_200mf(num_classes=1000):
    return RegNet(
        stem_channels=32,
        block_channels=[24, 56, 152, 368],
        num_blocks=[1, 1, 4, 7],
        num_classes=num_classes,
        group_width=8,
    )

# Example usage
if __name__ == "__main__":
    model = regnet_x_200mf(num_classes=10)  # CIFAR-10 has 10 classes
    print(model)

    inputs = torch.randn(1, 3, 224, 224)  # 1 sample, 3 channels (RGB), 224x224 image size
    outputs = model(inputs)
    print(outputs.shape)  # Should output torch.Size([1, 10])
