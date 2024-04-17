import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from modules import *


class DiffusionModel(pl.LightningModule):
    """
    A PyTorch Lightning module representing the Diffusion Model.

    Args:
        in_size (int): The input size of the model.
        t_range (int): The range of time steps.
        img_depth (int): The number of channels in the input images. -> usually 3 
    """

    def __init__(self, in_size, t_range, img_depth):
        super().__init__()
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.in_size = in_size

        self.unet = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=img_depth)

    def forward(self, x, t):
        """
        Forward pass of the Diffusion Model.

        Args:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The time step tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.unet(x, t)

    def beta(self, t):
        """
        Calculate the beta value based on the given time step.

        Args:
            t (int): The time step.

        Returns:
            float: The beta value.
        """
        # Just a simple linear interpolation between beta_small and beta_large based on t
        return self.beta_small + (t / self.t_range) * (self.beta_large - self.beta_small)

    def alpha(self, t):
        """
        Calculate the alpha value based on the given time step.

        Args:
            t (int): The time step.

        Returns:
            float: The alpha value.
        """
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        """
        Calculate the product of alphas from 0 to t.

        Args:
            t (int): The time step.

        Returns:
            float: The alpha bar value.
        """
        # Product of alphas from 0 to t
        return math.prod([self.alpha(j) for j in range(t)])

    def get_loss(self, batch, batch_idx):
        """
        Calculate the loss for the given batch.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        # Get a random time step for each image in the batch
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
        noise_imgs = []
        # Generate noise, one for each image in the batch
        epsilons = torch.randn(batch.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        # Run the noisy images through the U-Net, to get the predicted noise
        e_hat = self.forward(noise_imgs, ts)
        # Calculate the loss, that is, the MSE between the predicted noise and the actual noise
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss

    def denoise_sample(self, x, t):
        """
        Perform the denoising step for the given input and time step.

        Args:
            x (torch.Tensor): The input tensor.
            t (int): The time step.

        Returns:
            torch.Tensor: The denoised tensor.
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape)
            else:
                z = 0
            # Get the predicted noise from the U-Net
            e_hat = self.forward(x, t.view(1).repeat(x.shape[0]))
            # Perform the denoising step to take the image from t to t-1
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

    def training_step(self, batch, batch_idx):
        """
        Training step for the Diffusion Model.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        loss = self.get_loss(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the Diffusion Model.

        Args:
            batch (torch.Tensor): The input batch.
            batch_idx (int): The index of the batch.
        """
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss)
        return

    def configure_optimizers(self):
        """
        Configure the optimizer for the Diffusion Model.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
