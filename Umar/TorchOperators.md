## Torch Operators 

### Chunk

The [`torch.chunk`](https://pytorch.org/docs/stable/generated/torch.chunk.html) function splits a tensor into a specific number of chunks. The function takes two arguments: the tensor to split and the number of chunks to split it into. The function returns a tuple of tensors, each containing a chunk of the original tensor.

Example:  

Input Tensor: $ T \in \mathbb{R}^{B \times Seq_{Len} \times Dim}$ 

Output Tensors: $ q, k, v \in \mathbb{R}^{B \times Seq_{Len} \times Dim / n}$
```python
import torch
 # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
x = torch.randn(3, 20, 9)
q, k, v = x.chunk(3, dim=-1)
# q.shape, k.shape, v.shape -> torch.Size([3, 20, 3]), torch.Size([3, 20, 3]), torch.Size([3, 20, 3])
```

### View 

The [`torch.view`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) function reshapes a tensor to a new shape. The function takes a single argument, the new shape of the tensor. The function returns a new tensor with the specified shape. Returns a new tensor with the same data as the self tensor but of a different shape. The returned tensor shares the same data and must have the same number of elements, but may have a different size.

Example:

```python
import torch
# (Batch_Size, Seq_Len, Dim) -> (Batch_Size * Seq_Len, Dim)
x = torch.randn(3, 20, 9)
x = x.view(x, torch.Size([60, 9])) # or x.view(60, 9)
# x.shape -> torch.Size([60, 9])
```

### Ones_Like 

The [`torch.ones_like`](https://pytorch.org/docs/stable/generated/torch.ones_like.html) function creates a tensor of ones with the same shape as the input tensor. The function takes a single argument, the input tensor, and returns a new tensor of ones with the same shape as the input tensor. 

Example:

```python
import torch
# (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
x = torch.randn(3, 20, 9)
y = torch.ones_like(x)
# y.shape -> torch.Size([3, 20, 9])
```

### Triu

The [`torch.triu`](https://pytorch.org/docs/stable/generated/torch.triu.html) function returns the upper triangular part of a matrix. The function takes two arguments: the input tensor and the diagonal offset. The function returns a new tensor containing the upper triangular part of the input tensor. The diagonal and below are zeroed out.

Example:

```python
import torch
# (Batch_Size, Seq_Len, Seq_Len) -> (Batch_Size, Seq_Len, Seq_Len)
x = torch.randn(3, 20, 20)
y = torch.triu(x, diagonal=0)  # diagonal=0 means the main diagonal
# y.shape -> torch.Size([3, 20, 20])
```

