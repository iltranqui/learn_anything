import transformers
import datasets
import torch

import matplotlib.pyplot as plt

def plot_first_10_label(dataset):
    """Plots the 1st 10 labels of HuffingFace Dataset for Semantic Segmentation

    Args:
        dataset (datasets.dataset_dict.DatasetDict): _description_
    """
    # Create a figure with 2 rows and 5 columns for plotting 10 images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop over the first 10 images in the dataset and plot them
    for i in range(10):
        image = dataset["train"][i]['label']
    # label = dataset["train"][i]["label"]  # Assuming label information is available
        axes[i].imshow(image, cmap='gray')  # Assuming grayscale images
        #axes[i].set_title(f'Image {i+1}\nLabel: {label}')  # Set title with image index and label
        axes[i].axis('off')  # Turn off axis

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    # Example usage:
    # dataset is your dataset containing "train" data
    #plot_first_10_label(dataset)

def plot_first_10_images(dataset):
    """Plots the 1st 10 images of HuffingFace Dataset for Semantic Segmentation

    Args:
        dataset (datasets.dataset_dict.DatasetDict): _description_
    """
    # Create a figure with 2 rows and 5 columns for plotting 10 images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop over the first 10 images in the dataset and plot them
    for i in range(10):
        image = dataset["train"][i]["image"]
    # label = dataset["train"][i]["label"]  # Assuming label information is available
        axes[i].imshow(image, cmap='gray')  # Assuming grayscale images
        #axes[i].set_title(f'Image {i+1}\nLabel: {label}')  # Set title with image index and label
        axes[i].axis('off')  # Turn off axis

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    # Example usage:
    # dataset is your dataset containing "train" data
    #plot_first_10_images(dataset)

def plot_image_numpy(image):
    """
    Plot a single RGB image.

    Args:
    image (numpy.ndarray or torch.Tensor): The image data with shape (3, n, n) or (n, n, 3).
    """
    if isinstance(image, torch.Tensor):
        # Convert tensor to numpy array
        image = image.numpy()
    
    # Check if image shape is (3, n, n)
    if image.shape[0] == 3:
        # Transpose the image data to (n, n, 3) for matplotlib
        image = image.transpose(1, 2, 0)

    # Plot the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def plot_mask_numpy(mask):
    """
    Plot a single mask image.

    Args:
    mask (numpy.ndarray or torch.Tensor): The mask data with shape (1, n, n) or (n, n, 1).
    """
    if isinstance(mask, torch.Tensor):
        # Convert tensor to numpy array
        mask = mask.numpy()
    
    # Check if mask shape is (1, n, n)
    if mask.shape[0] == 1:
        # Squeeze the mask to remove singleton dimension
        mask = np.squeeze(mask)
    
    # Plot the mask
    plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.show()
