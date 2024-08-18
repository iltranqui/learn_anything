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
            torch.arange(self.num_positions).expand((1, -1)) # expand the tensor to the number of patches
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