# Flash Attention

## Standard Attention Implementation

> [!NOTE]
> Work in progress

Given the **query** and **key** and **value** matrices, $\in \mathbb{R}^{(N \times d )}$, where N is the sequence length and $d$ is the head dimension, we want the output to be $\in \mathbb{R}^{(N \times d )}$. 

$$ 
\bold S = QK^T \in \mathbb{R}^{(N \times N)}, P = softmax(\bold S) \in \mathbb{R}^{(N \times N)}, O = PV \in \mathbb{R}^{(N \times d)}
$$

where the attention is applied row-wise. First a matrix multiplication, then a softmax and then a matrix multiplication. This is done within the memory. 

The standard attention is implemented in the following pseudocode: 

**Algorithm 0** Standard Attention Implementation
----------------------------------------------
**Require**: Matrices $\bold Q, K, V \in \mathbb{ℝ}^{N \times d}$ in HBM
----------------------------------------------
1. Load **Q**, **K** by blocks from HBM, compute $\bold S = QK^T$, write **S** to HBM.
2. Read **S** from HBM, compute **P = softmax(S)**, write **P** to HBM.
3. Load **P** and **V** by blocks from HBM, compute **O = PV**, write **O** to HBM.
4. Return **O**.
----------------------------------------------

Flash Attention is only concerned with the softmax operation, and not the Matrix Multiplication. The MATMUL has already been optimized to death. 

## Safe Softmax 

Placing these matrices as reference; $ \bold S = QK^T \in \mathbb{R}^{(N \times N)}, P = softmax(\bold S) \in \mathbb{R}^{(N \times N)}, O = PV \in \mathbb{R}^{(N \times d)}$. We make the softmax operation per row obtaining the following below matrices:  

$$
\begin{pmatrix}
q_1^Tk_1 & q_1^Tk_2 & q_1^Tk_3 & q_1^Tk_4 & q_1^Tk_5 \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
\vdots & \ddots & \ddots & \ddots & \vdots \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
q_5^Tk_1 & q_5^Tk_2 & q_5^Tk_3 & q_5^Tk_4 & q_5^Tk_5
\end{pmatrix}
\quad \xrightarrow{\text{softmax}} \quad
\begin{pmatrix}
0.01 & 0.05 & 0.50 & 0.40 & 0.01 \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
\vdots & \ddots & \ddots & \ddots & \vdots \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
0.01 & 0.05 & 0.50 & 0.38 & 0.03    
\end{pmatrix}
$$

The softmax opeartion is defined as:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$
but this operation is numerically unstable. The softmax operation is numerically unstable because the exponential function grows very quickly, so to avoid this numberical instablility we add to the exponential function a constant $c$ such that the maximum value of the vector is subtracted from the vector, in log format like so: $c = \displaystyle max_i(x_i )$, then $e^{x_i - k}$, this is known as the log softmax. rendering the above softmax operation as: 
$$
softmax(x_i) = \frac{e^{x_i - k}}{\sum_{j=1}^{N} e^{x_j - k}} \text{ where } k = \displaystyle -log(c)
$$ rendering the above softmax operation numerically stable for float32, the most used dtype in deep learning.

Now we are entering problems of time complexity and memory complexity. We have 3 steps:
1.  Find the maximum value of the vector, we need to scan the entire row, 
    - a time complexity of $O(N)$
    - a memory complexity of $O(N)$. This is not ideal for large matrices.
2. Subtract the maximum value from the vector and Normalize the vector, 
    - a time complexity of $O(N)$
    - a memory complexity of $O(N)$. This is not ideal for large matrices.
3. Apply the softmax operation, 
    - a time complexity of $O(N)$
    - a memory complexity of $O(N)$. This is not ideal for large matrices.

These operations need to be more sequentially, *not an ideal sitatuation for GPUs*, which are designed to do parallel operations. and for larger matrices. *just immagine the context window of the latest LLLs*. Since we need this for the attention mechanism, we will repeat this search for 3 times, for the query, key and value matrices.
