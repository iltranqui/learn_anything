import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import get_model
from dataset import get_dataloaders

# Load hyperparameters from config file
with open('config.json', 'r') as f:
    config = json.load(f)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, val_loader, test_loader = get_dataloaders(config['batch_size'])

# Load model
# Load model
model = get_model(
    config['model_name'], 
    config['num_classes'],
    num_filters=config['num_filters'],
    dropout_rate=config['dropout_rate']
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Training loop
# Training loop with validation after each epoch
def train(num_epochs, model, train_loader, val_loader, criterion, optimizer):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training phase
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # Validate after each epoch
        validate(model, val_loader, criterion)

# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# Testing loop
def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = correct / total
        print(f'Test Accuracy: {test_acc:.4f}')

# Run training, validation after each epoch, and testing after all epochs
if __name__ == '__main__':
    # Train the model and validate every epoch
    train(config['num_epochs'], model, train_loader, val_loader, criterion, optimizer)
    # Test the model after training
    test(model, test_loader)