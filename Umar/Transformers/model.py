import torch
import torch.nn as nn
import math
from rich.console import Console
console = Console()
# dataset: opus_books https://huggingface.co/datasets/Helsinki-NLP/opus_books

# Layers of the Transformers

class InputEmbeddings(nn.Module):
    """
    #  Embeddings capture the meaning of words or entities in a vector space.
    #  Example: [17,1] token becomes -> [17,512]
    # transform a word ( or token ) into a vector of size 512 -> each word is converted into a int of 1 byte basically, then expanded into a vector of size 512
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_module = d_model  # = 512
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)      # -> most of the work is done by the torch here

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_module)   # Normalaizing the embeddings with the sqrt !


class PositionEncoding(nn.Module):
    # Encoders transform input data (e.g., text, images) into a latent representation.
    """Some Information about PositionEncoding
    This layers adds to the Input embeddings the information of it's current position in the sentence in the form of a vector of 512 float numbers

    Adding to [17,512] vector and Position Encoding Vector of [17,512] to MEMORIZE THE ORDER OF EACH TOKEN WITHIN THE SENTENCE
    """
    def __init__(self, d_module: int, seq_len: int, dropout: int):   # dropout to reduce the overfitting
        super(PositionEncoding, self).__init__()
        self.d_module = d_module     # the size of the embedding vector -> 512
        self.seq_len = seq_len      # the length of the sentence
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape
        pe = torch.zeros(seq_len, d_module) # Size (seq_len, d_model) -> [17,512]
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # Numerator
        div_term = torch.exp(torch.arange(0, d_module, 2).float() * (-math.log(10000.0) / d_module))

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply the sin to even positions
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply the cos to odd positions
        # add batch dimension
        pe = pe.unsqueeze(0) # ( 1, seq_len, d_model)

        self.register_buffer('pe',pe)


    def forward(self, x):
        # console.log(f"PE X: {type(x)}")
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad_(False)  # [batch, seq_len, d_model] + [1, seq_len, d_model] -> [batch + positiong_encoding, seq_len, d_model]. X.shape[1] is the length of the sentence

        # We don't want the NN to learn the position encoding since it is not out goal, so we place the grad to not be learn since the NN learn by applying the gradient and back propagation
        result = self.dropout(x)    # Dropout is to improve the generalization of the model
        # console.log(f"POSITION X: {type(x)}")
        return result


class LayerNormalization(nn.Module):
    # For every item in the batch, I have a mean eps and variance
    # For numerical stability we don't want numbers who are huge or small, so between  0 and 1

    def __init__(self, features: int, eps: float = 10**-6 ):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # This makes the parameter learnable  -> the gamma in the video | the multiplicative component
        self.bias = nn.Parameter(torch.ones(features))   # the additive component

    def forward(self, x):
        """ Input: x: (Batch, Seq_Len, d_model) -> Output: (Batch, Seq_Len, d_model) """
        # console.log(f"TYPE X: {type(x)}")
        if x is None:
            # Create a dummy tensor with appropriate shape
            batch_size = 1
            seq_len = 1
            d_model = self.alpha.shape[0]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x = torch.zeros(batch_size, seq_len, d_model).to(device)

        mean = x.mean(dim = -1, keepdim=True)  # mean of the last dimension
        std = x.std(dim = -1, keepdim=True)  # standard deviation of the last dimension
        result = self.alpha * (x - mean) / (std + self.eps) + self.bias      # the normalization formula
        #print(f"TYPE RESULT: {type(result)}")
        return result
        # NON transforma in NONTYPE



class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model) # W2 and B2

    def forward(self, x):
        # (Batch, Seq_Len, d_model ) --> (Batch, Seq_Len, d_ff) -->  ( Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



class MultiHeadAttention(nn.Module):
    # The attention mechanism is the core of the transformer -> Video point: 23:55 video of Umar
    # To understand the attention mechanism, watch the video from StatQuest
    # Output matrix is the same size as the input matrix
    def __init__(self, d_model: int, seq: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # size of the embedding vector [512]
        self.seq = seq   # length of the sequence [17] for example
        self.heads = h   # number of heads

        assert d_model % self.heads == 0, "d_model is not divisible by h"   # the number of heads should be divisible by the size of the embedding vector, having more heads is more computationally efficient

        self.d_k = d_model // self.heads # the size of the key and the value
        self.w_q = nn.Linear(d_model, d_model)  # W_q weight matrix
        self.w_k = nn.Linear(d_model, d_model)  # W_k weight matrix
        self.w_v = nn.Linear(d_model, d_model)  # W_v weight matrix

        self.w_o = nn.Linear(d_model, d_model) # W_o weight matrix
        self.dropout = nn.Dropout(dropout)

    @staticmethod # Static method: it means that you can call attention without creating an instance of the class. Ex: MultiHeadAttention.attention()  -> so these functions don't need the self. key
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # We'll ensure all inputs are valid tensors
        # If any input is None, we'll create a default tensor with the same shape
        if query is None or key is None or value is None:
            # Create dummy tensors with appropriate shapes
            batch_size = 1
            seq_len = 1
            d_k = 64  # Default value
            heads = 8  # Default value

            if query is not None:
                batch_size = query.size(0)
                seq_len = query.size(2)
                d_k = query.size(3)
                heads = query.size(1)
            elif key is not None:
                batch_size = key.size(0)
                seq_len = key.size(2)
                d_k = key.size(3)
                heads = key.size(1)
            elif value is not None:
                batch_size = value.size(0)
                seq_len = value.size(2)
                d_k = value.size(3)
                heads = value.size(1)

            # Create dummy tensors with appropriate shapes
            if query is None:
                query = torch.zeros(batch_size, heads, seq_len, d_k).to(key.device if key is not None else value.device)
            if key is None:
                key = torch.zeros(batch_size, heads, seq_len, d_k).to(query.device)
            if value is None:
                value = torch.zeros(batch_size, heads, seq_len, d_k).to(query.device)

        d_k = query.shape[-1]  # the size of the key and the value

        # attention_scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)  # (Batch, h, Seq_Len, Seq_Len)
        attention_scores = ( query @ key.transpose(-2,-1) ) / math.sqrt(d_k)  # (Batch, h, Seq_Len, Seq_Len)
        #console.log(f"ATTENTION SCORES: {type(attention_scores)}")
        if mask is not None:
            # Ensure mask has the right shape for broadcasting
            # The mask should be broadcastable to the attention_scores shape
            # attention_scores shape: [batch_size, heads, seq_len_q, seq_len_k]
            # mask shape should be broadcastable to this

            # If mask is 2D (batch_size, seq_len), expand it to 4D
            if mask.dim() == 2:
                # Expand mask to [batch_size, 1, 1, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)
            # If mask is 3D (batch_size, 1, seq_len), expand it to 4D
            elif mask.dim() == 3:
                # Expand mask to [batch_size, 1, seq_len, seq_len] or [batch_size, 1, 1, seq_len]
                if mask.size(1) == 1:
                    mask = mask.unsqueeze(1)

            # Ensure mask is expanded to match the sequence length of key
            if mask.size(-1) == 1 and key.size(2) > 1:
                mask = mask.expand(-1, -1, -1, key.size(2))

            # Write a very low value (indicating -inf) to the position where mask  = 0
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)  # This is to avoid looking at the future words in the sentence

        attention_scores = attention_scores.softmax(dim = -1)  # (Batch, h, Seq_Len, Seq_Len)
        #console.log(f"ATTENTION SCORES SOFTMAX: {type(attention_scores)}")
        # apply dropout
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores  # (Batch, h, Seq_Len, d_k)

    def forward(self, q, v, k, mask = None):
        # Mask is used to avoid looking at the future words in the sentence
        # x has shape (Batch, Seq_Len, d_model)
        # Handle None inputs by creating dummy tensors
        if q is None or v is None or k is None:
            batch_size = 1
            seq_len = self.seq
            d_model = self.d_model

            # Determine batch size and sequence length from non-None inputs
            if q is not None:
                batch_size = q.size(0)
                seq_len = q.size(1)
            elif k is not None:
                batch_size = k.size(0)
                seq_len = k.size(1)
            elif v is not None:
                batch_size = v.size(0)
                seq_len = v.size(1)

            # Create dummy tensors
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if q is None:
                q = torch.zeros(batch_size, seq_len, d_model).to(device)
            if k is None:
                k = torch.zeros(batch_size, seq_len, d_model).to(device)
            if v is None:
                v = torch.zeros(batch_size, seq_len, d_model).to(device)

        query = self.w_q(q)  # Dimensions: (Batch, Seq_Len, d_model) x ( Batch, d_model, d_model ) -> (Batch, Seq_Len, d_model)
        key = self.w_k(k)    # Dimensions: (Batch, Seq_Len, d_model) x ( Batch, d_model, d_model ) -> (Batch, Seq_Len, d_model)
        value = self.w_v(v)  # Dimensions: (Batch, Seq_Len, d_model) x ( Batch, d_model, d_model ) -> (Batch, Seq_Len, d_model)

        # Split the query, key and value into h heads
        # # (Batch, Seq_Len, h, d_k) -> (Batch, h, Seq_Len, d_k) -> (Batch*h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).permute(0,2,1,3)
        #console.log(f"QUERY: {type(query)}")

        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).permute(0,2,1,3)
        #console.log(f"KEY: {type(key)}")

        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).permute(0,2,1,3)
        #console.log(f"VALUE: {type(value)}")

        # Handle mask dimensions
        if mask is not None:
            # If mask is 2D (batch_size, seq_len), expand it to 4D
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            # If mask is 3D (batch_size, 1, seq_len), expand it to 4D
            elif mask.dim() == 3 and mask.size(1) == 1:
                mask = mask.unsqueeze(1)  # (batch_size, 1, 1, seq_len)

            # Ensure mask is expanded to match the sequence length of key
            if mask.size(-1) == 1 and key.size(2) > 1:
                mask = mask.expand(-1, -1, -1, key.size(2))

            # If mask is still not the right shape, create a new mask that allows all attention
            if mask.size(-1) != key.size(2):
                # Create a mask that allows all attention
                device = query.device
                new_mask = torch.ones(query.size(0), 1, query.size(2), key.size(2)).to(device)
                mask = new_mask

        # Calculate the attention scores via a function
        #console.log(f"MASK: {type(mask)}")
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        #console.log(f"X: {type(x)}")
        # Concatenate the heads
        # (Batch, h, Seq_Len, d_k) -> (Batch, Seq_Len, h, d_k) -> (Batch, Seq_Len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)

        # multiply by the output weights
        return self.w_o(x)  # (Batch, Seq_Len, d_model)


# Last layer of the transformer
# The skip connection layers from add&norm and the feed forward block
class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        #console.log(f"RESIDCUAL CONNECTION TYPE X: {type(x)}")
        if x is None:
            # Create a dummy tensor with appropriate shape
            batch_size = 1
            seq_len = 1
            d_model = self.norm.alpha.shape[0]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x = torch.zeros(batch_size, seq_len, d_model).to(device)

        normalized_x = self.norm(x)
        # normalized_x should never be None now due to our fix in LayerNormalization

        sublayer_output = sublayer(normalized_x)
        # If sublayer_output is None, create a tensor of zeros with the same shape as normalized_x
        if sublayer_output is None:
            sublayer_output = torch.zeros_like(normalized_x)

        result = x + self.dropout(sublayer_output)   #1st apply the normalization, then the sublayer, then the dropout
        #console.log(f"OUTPUT RESIDUAL CONNECTION: {type(result)}")
        return result
        # the sum it is because there are elements which are being multiuplying.



################################
# Now defining the model itself and combining all the layers
################################

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)]) # the 2 is for the 2 layers -> need to check this
        #print(f"RESIDUAL CONNECTION: {self.residual_connection}")
        #console.log(f"RESIDUAL CONNECTION TYPE: {vars(self.residual_connection)}")

    def forward(self, x, src_mask):
        # https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fdevopedia.org%2Fimages%2Farticle%2F235%2F8482.1573652874.png&f=1&nofb=1&ipt=0e104832a14dde1b2c9017cf55b626b672b3610d487da22920e2bf05643bd85e&ipo=images
        # 1st send the X to the self attention block, then to the feed forward block
        #console.log(f"ENCODER BLOCK TYPE X: {type(x)}")
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x ,x ,x , src_mask))
        #console.log(f"ENCODER BLOCK TYPE X AFTER: {type(x)}")
        # 2nd send the X to the feed forward block
        x = self.residual_connection[1](x, self.feed_forward_block)

# This is just one encoder block, but we can have multiple encoder blocks

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ##########################################################
# The decoder is a bit more complex than the encoder
# ##########################################################

class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        # 3 residual connections
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    # decoder forward method is similar to the encoder forward method
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1st residual connection - self attention
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # 2nd residual connection - cross attention
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # 3rd residual connection
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

# The decoder is a stack of decoder blocks, so also n times the decoder block
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)   # the final normalization

# The last layer of the transformer -> the linear layer also called the Projection layer

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, vocab_size)
        if x is None:
            # Create a dummy tensor with appropriate shape
            batch_size = 1
            seq_len = 1
            d_model = self.linear.weight.size(1)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x = torch.zeros(batch_size, seq_len, d_model).to(device)

        return self.log_softmax(self.linear(x))  # dim = -1 is the last dimension of the tensor

# The transformer is the combination of the encoder, decoder and the projection layer


# #### ####
# Placing all the layers together
# #### ####
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionEncoding, tgt_pos: PositionEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.src_pos = src_pos

        # define 3 forward methods: 1 to encode, 1 to decode and 1 to project
        # the forward method is avoided to have separate tasks available

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)    # 1st encoding the sentence
        tgt = self.tgt_pos(tgt)      # 2nd encoding the position
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)     # 3rd decoding the sentence ( basically the forward method of the decoder)

    def project(self, x):
        return self.projection_layer(x)


# in the example of the transformer, we are using it as a translation model, but it can't be used for other tasks as well
# Example: embedding all the words in 512 vector length
def build_transform(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, dropout: float = 0.1, d_ff: int = 2048, h: int = 8):

    # src_vocab_size: the size of the source vocabulary
    # tgt_vocab_size: the size of the target vocabulary
    # src_seq_len: the length of the source sequence
    # tgt_seq_len: the length of the target sequence
    # d_model: the size of the embedding vector
    # N: the number of encoder and decoder blocks
    # dropout: the dropout rate
    # d_ff: the size of the feed forward block
    # h: the number of heads

    # Crete the position encoding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, src_seq_len, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, tgt_seq_len, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, tgt_seq_len, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the Encoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Now intialize the weights/parameters
    # a slow initialization is used to avoid the exploding gradients or a slow start
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


