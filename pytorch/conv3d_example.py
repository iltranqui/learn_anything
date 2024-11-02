import torch
import torch.nn as nn

# Define input parameters
batch_size = 1       # Example batch size
in_channels = 3      # Example number of input channels
depth, height, width = 8, 32, 32  # Dimensions of the 3D volume

# Create a random input tensor with the specified dimensions
input_tensor = torch.randn(batch_size, in_channels, depth, height, width)

# Define a Conv3D layer
out_channels = 6     # Number of output channels
kernel_size = (3, 3, 3)  # Kernel size for depth, height, width
conv3d_layer = nn.Conv3d(in_channels, out_channels, kernel_size)

# Apply the Conv3D layer to the input tensor
output_tensor = conv3d_layer(input_tensor)

# Print the output shape to verify the operation
print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a small neural network with a Conv3D layer and a Linear layer
class Small3DNN(nn.Module):
    def __init__(self):
        super(Small3DNN, self).__init__()
        
        # 3D convolutional layer
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=(3, 3, 3))
        
        # Fully connected layer
        # Output size will depend on the input tensor dimensions after conv3d, so we need to calculate it
        self.fc = nn.Linear(8 * 6 * 30 * 30, 10)  # Adjusted to the expected flattened size
    
    def forward(self, x):
        # Pass through Conv3D layer with ReLU activation
        x = F.relu(self.conv3d(x))
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Pass through the fully connected layer
        x = self.fc(x)
        return x

# Instantiate the neural network
model = Small3DNN()

# Create a random input tensor with dimensions (batch_size, in_channels, depth, height, width)
input_tensor = torch.randn(1, 3, 8, 32, 32)  # Batch size = 1, in_channels = 3, depth = 8, height = 32, width = 32

# Perform a forward pass through the model
output = model(input_tensor)

# Print the output shape
print("Output shape:", output.shape)
print("Output:", output)
