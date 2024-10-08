import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import get_model  # Assumed to have a model definition
from dataset import get_dataloaders  # Assumed to have a data loading function
from train import train, test  # Assumed to have training and testing functions
import json
import optuna


# Load hyperparameters from config file
with open('config.json', 'r') as f:
    config = json.load(f)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the objective function
def objective(trial):
    # Hyperparameters to tune
    #batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    num_filters = trial.suggest_categorical('num_filters', [16, 32, 64])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    num_epochs = 2  # Fixed number of epochs for simplicity
    batch_size = 2048  # Fixed batch size for simplicity

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size)

    # Load model
    model = get_model(
        config['model_name'], 
        config['num_classes'],
        num_filters=num_filters,
        dropout_rate=dropout_rate
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # MLflow experiment tracking
    with mlflow.start_run() as run:
        mlflow.log_params({
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_filters': num_filters,
            'dropout_rate': dropout_rate,
        })

        # Train the model and validate every epoch
        train(num_epochs, model, train_loader, val_loader, criterion, optimizer)
        # Test the model after training
        test_loss = test(model, val_loader)
        return test_loss

# Create and optimize the study
if __name__ == '__main__':
    # Set up MLflow experiment
    mlflow.set_experiment('Optuna_Hyperparameter_Optimization')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters and their validation accuracy
    print(f'Best hyperparameters: {study.best_params}')
    print(f'Best validation loss: {study.best_value:.4f}')
