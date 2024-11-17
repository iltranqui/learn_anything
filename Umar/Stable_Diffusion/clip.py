import torch 
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class CLIPEmbedding:
    """
    CLIPEmbedding is a neural network module for generating embeddings from a given vocabulary.
    It embeds the tokens and adds positional embeddings to them.

    **Attributes**
    -------------
    - **n_vocab** (int): The size of the vocabulary used.
    - **n_embed** (int): The dimensionality of the embedding vectors (standard is 512).
    - **n_tokens** (int): The number of tokens to be embedded.

    **Methods**
    -----------
    - **__init__(n_vocab: int, n_embed: int, n_tokens: int)**:
        Initializes the CLIPEmbedding with the given vocabulary size, embedding dimension, and number of tokens.

    - **forward(tokens: torch.LongTensor) -> torch.FloatTensor**:
        Performs the forward pass of the CLIPEmbedding module.

        - **Args**:
            - tokens (torch.LongTensor): The input tensor of shape (Batch_Size, Seq_Len).
        
        - **Returns**:
            - torch.FloatTensor: The output tensor of shape (Batch_Size, Seq_Len, Dim).
    """


    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)   # Token embeddings done by pytorch
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed)) # Positional embeddings done by us

    def forward(self, tokens):
        """
        Forward pass of the CLIPEmbedding module.
        Args:
            tokens (torch.LongTensor): The input tensor of shape (Batch_Size, Seq_Len).
        Returns:
            torch.FloatTensor: The output tensor of shape (Batch_Size, Seq_Len, Dim).
        """
        x = self.token_embedding(tokens)

        x += self.position_embedding

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        return x
    

class CLIPlayer(nn.Module):
    """
    CLIPlayer is a neural network module that consists of a self-attention layer followed by a feed-forward layer.
    THis represnts a single layer of the CLIP model. 

    
    **Attributes**
    -------------
    - **n_heads** (int): The number of attention heads.
    - **n_embed** (int): The dimensionality of the embedding vectors (standard is 512).

    **Methods**
    ----------
    - **forward(x: torch.FloatTensor) -> torch.FloatTensor**:
        Performs the forward pass of the CLIPlayer module.

        - **Args**:
            - x (torch.FloatTensor): The input tensor of shape (Batch_Size, Seq_Len, Dim).
        
        - **Returns**:
            - torch.FloatTensor: The output tensor of shape (Batch_Size, Seq_Len, Dim).

    """
    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads, n_embed)
        self.layernorm2 = nn.LayerNorm(n_embed)

        self.linear1 = nn.Linear(n_embed, 4 * n_embed)  # why 4 ? 
        self.linear2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x):
        """
        Forward pass of the CLIPlayer module.
        Args:
            x (torch.FloatTensor): The input tensor of shape (Batch_Size, Seq_Len, Dim).
        Returns:
            torch.FloatTensor: The output tensor of shape (Batch_Size, Seq_Len, Dim).
        """
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        residual = x

        # SELF ATTENTION

        x = self.layernorm1(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, casual_mask=True)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residual

        # FEED FORWARD
        residual = x

        x = self.layernorm2(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.linear1(x)

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = x * torch.sigmoid(1.702 * x) # QuickGELU activation function # Why 1.702 ? -> https://arxiv.org/abs/2003.10555 in the paper they just say it works better

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear2(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residual

        # (Batch_Size, Seq_Len, Dim)
        return x

    
"""
Skeleton of the CLIP model
"""
class CLIP(nn.Module):

    def __init__(self):
        super(CLIP, self).__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77) 

        self.layers = nn.Module([
            CLIPlayer(12, 768) for i in range(12)   # 12 layers of Transformers Layers
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # ( Batch_Size, Seq_LEn ) -> ( Batch_Size, Seq_Len, Dim )
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)

        # (Batch_Size, Seq_Len, Dim)
        return output
    
