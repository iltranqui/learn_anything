import torch
from torchvision import datasets, transforms

def get_dataloaders(batch_size, data_dir='./data'):
    # Define transforms
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),   # Resizing images to 224x224 only more complex models require this
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # Split test dataset into validation and test datasets
    val_size = 5000
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
    print(f'Number of training batches: {len(train_loader)}')
    print(f'Number of validation batches: {len(val_loader)}')
    print(f'Number of test batches: {len(test_loader)}')

    # Print shapes of a batch from each loader
    def print_batch_shapes(loader, loader_name):
        dataiter = iter(loader)
        images, labels = next(dataiter)
        print(f'Loader: {loader_name}')
        print(f'  Images batch shape: {images.shape}')
        print(f'  Labels batch shape: {labels.shape}')
    
    print_batch_shapes(train_loader, 'Train Loader')
    print_batch_shapes(val_loader, 'Validation Loader')
    print_batch_shapes(test_loader, 'Test Loader')