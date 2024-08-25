from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisualizationConfig:

    def __init__(self,
                 hidden_size: int = 768, # hidden size of the embeddings vector for Vit
                 intermediate_size: int = 3072, # size of layer for the linear feedforward network
                 num_hidden_layers: int = 12, # number of layers for the transformer ( but there are one on top of another ?? )
                 num_channels: int = 3, # number of channels for the input image
                 num_attention_heads: int = 12, # number of attention heads for the transformer
                 image_size: int = 224, # size of the image
                 patch_size: int = 16, # size of the patch 16*16
                 layer_norm_eps: float = 1e-12, # 
                 attention_dropout: float = 0.0, # dropout rate for the attention layer
                 num_image_tokens: int = 1, # how many image embeddings for each image
                 ** kwargs   # other arguments that are not used
                 ):

        super().__init__()  # super(). is used to call the __init__() of the parent class (nn.Module) and to access its properties.

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
# All the classes below need to extract different parameters from the config class above, which are shared among all the classes


# In LLM, a token can be a word, a sentence, a paragraph, or an image patch. It can ne even a sequence of words. 
class SiglipMLP(nn.Module):

    def __init__(self, config: SiglipVisualizationConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embedding_Size] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # Nonlinear activation function
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embedding_Size]
        hidden_states = self.fc2(hidden_states)
        # 2nd Activation function
        # hidden_states = nn.functional.gelu(hidden_states)
        return hidden_states
    

# HIGH LEVEL: This is the union of all the layers of the transformer
class SiglipEncoder(nn.Module):

    def __init__(self, config: SiglipVisualizationConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # inputs_embeds: [batch_size, num_patches, embed_dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


# HIGH LEVEL: This is a single layer of the transformer, which is composed of a multi-head attention layer and a feedforward layer
class SiglipAttention(nn.Module):

    # TLDR: the multi-head attention is a linear transformation of the input, which is then split into multiple heads. Each token is divided into smaller embeddings, like from 1024 = 8 * 128
    # Then the entire heads work in PARALLEL, and the output is concatenated and linearly transformed again. -> MORE HEADS, more parallelization, but more computation

    def __init__(self, config: SiglipVisualizationConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5  # Equivalment to 1/sqrt(head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Here we are going to implement the attention mechanism, qkv are treted independently
        # hidden_states: [batch_size, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.shape
        # query states: [batch_size, num_patches, embed_dim]
        query_states = self.q_proj(hidden_states)   # this is a linear transformation of the hidden states
        # key states: [batch_size, num_patches, embed_dim]
        key_states = self.k_proj(hidden_states)
        # value states: [batch_size, num_patches, embed_dim]
        value_states = self.v_proj(hidden_states)

        # We divide the query, key and value states into num_attention_heads -> Just grouping the dimensions  https://youtu.be/vAmKB7iPkWw?si=G-FwSrVg0RFe7vqt&t=6104
        # query_states: [batch_size, num_heads, num_patches, head_dim]
        # Ezample : (4, 1024) * (1024, 8, 128) = (4, 8, 128)
        # Ezample : (sequence_length, hidden_size) * (hidden_size, num_attention_heads, head_dim) = (sequence_length, num_attention_heads, head_dim) 
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # We transpose the query, key and value states to have the following shape: [batch_size, num_attention_heads, head_dim] so [4, 8, 128] so we can make the dot product of the qkv elements, so the self attention mecahnicms has the contect of the entire token
        # Calculate the attention using the formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V. attention_weights: [batch_size, num_attention_heads, num_patches, num_patches]
        # the attention mask is applied to the tokens of interest, so the ignored tokens, by setting the attention weights to inf, so that the softmax will be zero
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_attention_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should have the shape [batch_size, num_attention_heads, seq_len, seq_len],"
                f"but has {attn_weights.size()}"
            )
    
        # apply the softmax to the attention weights row wise, so that the sum of the weights is 1
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # apply dropout to the attention weights
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)  # way to reduce overfitting
        # multiply the attention weights by the value states
        # attn_output: [batch_size, num_attention_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if (attn_output.size() != (batch_size, self.num_attention_heads, seq_len, self.head_dim)):
            raise ValueError(
                f"Attention output should have the shape [batch_size, num_attention_heads, seq_len, head_dim],"
                f"but has {attn_output.size()}"
            )
        
        # Transpose the attn_output to have the following shape: [batch_size, seq_len, num_attention_heads, head_dim]
        # attn_output: [batch_size, num_patches, num_attention_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()    # contiguous() is used to make the memory contiguous -> the same way the class cv::Mat works in C++ XD 
        # attn_output: [batch_size, num_patches, embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # Concatenate the attention output of all heads into one single tensor
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights



class SiglipVisualEmbeddings(nn.Module):
    
    def __init__(self, config: SiglipVisualizationConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # why not use the openai embedding vector ? 
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size, # number also of filters
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid", # this indicates that no padding is added
        
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2   # number of patches in the image ->  224/16 for height and width, remember that there is no padding
        self.num_positions = self.num_patches # how many positional encodings do we need ? 
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim) # a vector of the same size as the embeddings
        # register_buffer is used to register the positional embeddings: they are not parameters of the model, but they are part of the model state
    
        self.register_buffer(   # registering the info of the positinal embeddings, so that they can be seen
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)), # expand the tensor to the number of patches
            persistent=False, 
        )

    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape  # [batch_size, num_channels, height, width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        x = self.patch_embeddings(pixel_values)  # [batch_size, hidden_size, num_patches]
        # [batch_size, hidden_size, num_patches] -> [batch_size, num_patches, Num_Patches] 
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, hidden_size]
        # Add postion embeddings to each patch. Each positional encoding is a vector of size `embed_dim`
        embeddings = embeddings + self.position_embeddings(self.position_ids)  
        # [batch_size, num_patches, embed_dim]
        return embeddings
        



class SiglipVisualTransformer(nn.Module):

    def __init__(self, config: SiglipVisualizationConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embedding = SiglipVisualEmbeddings(config) # 1st: extrtacting the patches from the image
        self.encoder = SiglipEncoder(config)   # 2nd: encoding the patches throguh a series of transformer layers
        self.post_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [batch_size, num_channels, height, width] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.embedding(pixel_values)   # Embedding of the patches 

        last_hidden_state = self.encoder(hidden_states)  # Encoding the patches through the transformer layers

        last_hidden_state = self.post_layer_norm(last_hidden_state)  # Layer normalization

        return last_hidden_state



class SiglipVisualModel(nn.Module): # nn.Module is the base class for all neural network modules in PyTorch

    def __init__(self, config: SiglipVisualizationConfig):
        super().__init__()   # initialize the parent class and the properties of the class above 
        self.config = config
        self.vision_model = SiglipVisualTransformer(config) # initialize the vision model with the config

    def forward(self, pixel_values: torch.Tensor) -> Tuple: # forward pass of the model, for inference and forward pass 
        # [batch_size, num_channels, image_size, image_size] -> [batch_size, num_patches, Embedding_size]
        # So each image is divided into a nmber of patches, which will be an of embeddings vector of size Embedding_size
        # Output: List of Embeddings for each image. 
        # OBS: using list instead of arras, since image can have different sizes
        return self.vision_model(pixel_values=pixel_values)