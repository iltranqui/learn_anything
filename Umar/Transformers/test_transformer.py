import pytest
import torch
import torch.nn as nn
from math import sqrt
from model import *

# Import the classes to test (assuming they're available from the transformer module)

# Define a pytest fixture to setup common parameters for the tests
@pytest.fixture
def setup_params():
    d_model = 512
    vocab_size = 1000
    seq_len = 10
    dropout = 0.1
    return d_model, vocab_size, seq_len, dropout

# Test for InputEmbeddings class
def test_input_embeddings(setup_params):
    d_model, vocab_size, seq_len, _ = setup_params
    input_embedding = InputEmbeddings(d_model, vocab_size)
    
    x = torch.randint(0, vocab_size, (2, seq_len))  # Create random input with batch_size=2 and sequence length=seq_len
    output = input_embedding(x)

    assert output.shape == (2, seq_len, d_model), f"Expected shape (2, {seq_len}, {d_model}) but got {output.shape}"
    assert torch.is_tensor(output), "Output should be a tensor"
    assert torch.all(output != 0), "Embeddings should not be zero"

# Test for PositionEncoding class
def test_position_encoding(setup_params):
    d_model, _, seq_len, dropout = setup_params
    position_encoding = PositionEncoding(d_model, seq_len, dropout)
    
    x = torch.zeros(2, seq_len, d_model)  # Create zeroed input
    output = position_encoding(x)

    assert output.shape == (2, seq_len, d_model), f"Expected shape (2, {seq_len}, {d_model}) but got {output.shape}"
    assert torch.is_tensor(output), "Output should be a tensor"
    assert not torch.allclose(output, x), "Position encoding should modify input"

# Test for LayerNormalization class
def test_layer_normalization(setup_params):
    d_model, _, seq_len, _ = setup_params
    layer_norm = LayerNormalization(d_model)

    x = torch.randn(2, seq_len, d_model)  # Random input
    output = layer_norm(x)

    assert output.shape == (2, seq_len, d_model), f"Expected shape (2, {seq_len}, {d_model}) but got {output.shape}"
    assert torch.is_tensor(output), "Output should be a tensor"
    assert not torch.allclose(output, x), "Layer normalization should modify input"

# Test for FeedForwardBlock class
def test_feed_forward_block(setup_params):
    d_model, _, seq_len, dropout = setup_params
    d_ff = 2048  # Set feed-forward size

    ff_block = FeedForwardBlock(d_model, d_ff, dropout)
    x = torch.randn(2, seq_len, d_model)  # Random input

    output = ff_block(x)
    
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert isinstance(x, torch.Tensor), "Input should be a tensor"
    assert output.shape == (2, seq_len, d_model), f"Expected shape (2, {seq_len}, {d_model}) but got {output.shape}"
    assert torch.is_tensor(output), "Output should be a tensor"

# Test for MultiHeadAttention class
def test_multihead_attention(setup_params):
    d_model, _, seq_len, dropout = setup_params
    heads = 8

    mha = MultiHeadAttention(d_model, seq_len, heads, dropout)
    q = k = v = torch.randn(2, seq_len, d_model)  # Random query, key, and value

    output = mha(q, k, v)

    assert output.shape == (2, seq_len, d_model), f"Expected shape (2, {seq_len}, {d_model}) but got {output.shape}"
    assert torch.is_tensor(output), "Output should be a tensor"

# Test for ResidualConnection class
def test_residual_connection(setup_params):
    d_model, _, seq_len, dropout = setup_params

    residual = ResidualConnection(d_model, dropout)
    x = torch.randn(2, seq_len, d_model)  # Random input
    sublayer = lambda x: x * 2  # Dummy sublayer operation (for testing)

    output = residual(x, sublayer)

    assert output.shape == (2, seq_len, d_model), f"Expected shape (2, {seq_len}, {d_model}) but got {output.shape}"
    assert torch.is_tensor(output), "Output should be a tensor"

