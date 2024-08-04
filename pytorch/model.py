import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN_simple(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN_simple, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(model_name, num_classes=10, conv_layers=[(32, 3), (64, 3)], fc_layers=[512], dropout_rate=0.5, **kwargs):
    if model_name == 'simple_cnn_simple':
        model = SimpleCNN_simple(num_classes=num_classes)
        return model
    else:
        raise ValueError("Model not supported.")
    
if __name__ == '__main__':
    # Generate random input
    input_size = (64, 3, 32, 32)
    input_tensor = torch.randn(input_size)

    model = get_model('simple_cnn_simple')
    output = model(input_tensor)
    print(output.shape)