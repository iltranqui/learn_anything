import torch.nn as nn

# Here i can change the size and shape of the architecture without messing with the main code ! 
# MINST DATASET ! 
# INput ( 1 , 28 , 28 ) 
# Output ( 1, 10 )

def experiment(hidden_size, num_classes, dims):
    channels, width, height = dims
    hidden_size = 256

    experiment = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels * width * height, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, num_classes),
    )

    return experiment