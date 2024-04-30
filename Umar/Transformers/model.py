import torch
import torch.nn as nn
import math

# Layers of the Transformers 

class InputEmbeddings(nn.Module):
    
    # transform a word into a vector of size 512 -> each word is converted into a int of 1 byte basically 
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_module = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)      # -> most of the work is done by the torch here

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_module)
    

class PositionEncoding(nn.Module):
    """Some Information about PositionEncoding
    This layers adds to the Input embeddings the information of it's current position in the sentence in the form of a vector of 512 float numbers
    """
    def __init__(self, d_module: int, seq_len: int, dropout: int):   # dropout to reduce the overfitting
        super(PositionEncoding, self).__init__()
        self.d_module = d_module
        self.seq_len = seq_len      # the length of the sentence
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape 
        pe = torch.zeros(seq_len, d_module)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # Numerator 
        div_term = torch.exp(torch.arange(0, d_module, 2).float() * (-math.log(10000.0) / d_module))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # add batch dimension
        pe = pe.unsqueeze(0) # ( 1, seq_len, d_model)

        self.register_buffer('pe',pe)


    def forward(self, x):
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad_(False)  # We don't want the NN to learn the position encoding since it is not out goal, so we place the grad to not be learn since the NN learn by applying the gradient and backpropagation
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    # For every item in the batch, I have a mean eps and variance 
    # For numerical stablility we don't want numbers who are huge or small, so between  0 and 1 

    def __init__(self, eps: float = 10**-6 ):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # This makes the parameter learnable  -> the gamma in the video | the multiplicative componet 
        self.bias = nn.Parameter(torch.ones(1))   # the additive component 

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias    
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) # W1 and B1 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model) # W2 and B2

    def forward(self, x):
        # (Batch, Seq_Len, d_model ) --> (Batch, Seq_Len, d_ff) -->  ( Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
