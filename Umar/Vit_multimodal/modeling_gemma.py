import torch 
from torch import nn
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple, Union, Iterable
from torch.nn import CrossEntropyLoss, MSELoss
import math 
from modeling_siglip import SiglipVisualizationConfig, SiglipVisualModel



class GemmaConfig():

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
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
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


class KVCache():

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

# The FeedForward Element of the Gemma LLM model
class GemmaMLP(nn.Module):  
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    # TODO: Understand the forward pass of Gemma
    def forward(self, x):
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))   
    

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

    # TO DO: Understand the merge_input_ids_with_image_features
    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        _, _, embed_dim = image_features.shape                      #image_features: the features of the image
        batch_size, sequence_length = input_ids.shape               #INput_ids: the position of each token in the tensor, sequence_length is the number of tokens 
        dtype, device = inputs_embeds.dtype, inputs_embeds.device   #inputs_embeds: the embeddings of the tokens

        # Scale the feature - Shape: [Batch_Size, Seq_Len, Hidden_Size] 
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
    
        # INitialize the final embedding tensor, starting from a zero one. 
        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        
        #@ Following: logic to distinguish between text tokens, image tokens and padding tokens 
        #@ input_ids: the position of each token in the tensor, which also descirbe if the token is a text token, an image token or a padding token
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### CREATE THE ATTENTION MASK ####

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids
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
    