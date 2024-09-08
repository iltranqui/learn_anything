import torch 
from torch import nn
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple, Union, Iterable
from torch.nn import CrossEntropyLoss, MSELoss
import math 
from modeling_siglip import SiglipVisualizationConfig, SiglipVisualModel



class GennaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,  ## FF layers 
        num_hidden_layers,  ## number of consecutive layers
        num_attention_heads,  
        num_key_values_heads,  # number of heads for the key and value, since they will be different
        head_dim=256,  # how many dimensions per head
        max_position_embeddings=8192,
        rms_norm_eps = 1e-6,
        rope_theta = 10000.0,
        attention_bias = False,
        attention_dropout=0.1,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_values_heads = num_key_values_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id




class PaliGemmaConfig():
    
    def __init__(
        self,
        vision_config: None,  # SiglipVisualizationConfig
        text_config: None,  # GemmaForCasualLM
        index_ignore: -100,  
        image_token_index: 256000,
        vocab_size: int = 257152,
        projection_dim: int = 2048,
        hidden_size: int = 2048,
        pad_token_id: int = None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = index_ignore
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.vision_config = vision_config
        self.is_enocder_decoder = False

        self.vision_config = SiglipVisualModel(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaForCasualLM(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


# PaliGemmaConfig is a class that contains the configuration of the model and HIGH level parameters
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisualModel(config.Vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCasualLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):   # tie weights: technique to reduce the number of parameters in a model by using the same weight matrix for multiple layers. Basically is the inverse operation of the embedding layer https://youtu.be/vAmKB7iPkWw?si=iYvPhTdDzvn0L3p3&t=9978
        self.language_model.tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        
        assert torch.all(attention_mask == 1), "input cannot be padded"

        # 1. Extra the input embeddings
        # shape (batch_size, sequence_length, hidden_size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)  # image tokens and text tokens are embedded in the same space
        
        # 2. Merge text and image embeddings
        # shape (batch_size, Channels, heihgt, width) -> (batch_size, sequence_length, embedding_size)
        selected_image_features = self.vision_tower(pixel_values.to(input_embeds.dtype))

        # 3. image features 
        # shape (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, hidden_size)
        image_features = self.multi_modal_projector(selected_image_features)


        # 4. Merge text and image features
        # shape (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, hidden_size)
        # inmage_feature: input of the iamge 
        # input_embeds: input of the text, contains text tokens and placeholder tokens
        # input_ids: input of the text

        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)

        # 5. Forward pass through the llm model with all the tokens
        outputs = self.language_model(
            attention_mask=attention_mask,
            input_ids=position_ids,
            inputs_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        return outputs
    