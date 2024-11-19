import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import SelfAttention, CrossAttention

"""
Diffusione model -> a modified UNET 
"""


class TimeEmbedding(nn.Module):
    """
    TEnsor to keep track of the added noise at various steps.
    """

    def __init__(self, n_embd: int):
        super().__init__(self):
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, time: torch.Tensor):
        """
        Forward pass of the TimeEmbedding module.
        Args:
            time (torch.Tensor): The input tensor of shape (1, 320).
        Returns:
            torch.Tensor: The output tensor of shape (1, 1280).
        """
        # x: (1, 320)
        time = self.linear_1(time)

        time = F.silu(time)
        
        time = self.linear_2(time)

        # (1, 1280)
        return time

class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
            return x


class UNET(nn.Module):

    def __init__():
        super().__init__()

        self.encoders = nn.Module([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1), 
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, padding=1), 
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640), kernel_size=3, padding=1), 
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        ) 

        self.decoders = nn.ModuleList([
            SwitchSequential(
                """
                Expect double the size, since we are considering the skip connections. 
                """
                UNET_ResidualBlock(2560, 1280), 
            )

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

        


class Diffusion(nn.Module):
    """
    Diffusion is a neural network module that consists of a UNET model with an additional time embedding layer.

    **Attributes**
    -------------
    - **time_embedding** (nn.Embedding): The time embedding layer for the scheduler and to keep track of the .
    - **unet** (UNET): The UNET model.
    - **final** (UNET_OutputLayer): The output layer of the UNET model.
    """

    def __init__(self):
        self.time_embedding = nn.Embedding(100, 512)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        """
        Forward pass of the Diffusion module.
        Args:
            latent (torch.Tensor): The latent space tensor of shape (Batch_Size, 4, Height / 8, Width / 8)
            context (torch.Tensor): The context tensor with the prompt of shape (Batch_Size, Seq_Len, Dim).
            time (torch.Tensor): The tensor keep track of the added noise at various steps (1, 320)
        Returns:
            torch.Tensor: The output tensor of shape (Batch_Size, 4).
        """

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        """
        last layer of the UNET model must bring use to the original input, it DOENS'T have to give back the original input, that is the job of the UNET_OutputLayer
        The UNET model predicts the next step in the diffusion process, the UNET_OutputLayer will predict the final output
        """
        output = self.unet(latent)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        output = self.final(output)

        # (Batch_Size, 4, Height / 8, Width / 8) 
        return output
    