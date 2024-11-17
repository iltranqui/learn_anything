# Flash Attention

## Standard Attention Implementation

Given the **query** and **key** and **value** matrices, $\in \mathbb{R}^{(N \times d )}$, where N is the sequence lenth and $d$ is the head dimension, we want the output to be $\in \mathbb{R}^{(N \times d )}$. 

$$ 
\bold S = QK^T \in \mathbb{R}^{(N \times N)}, P = softmax(\bold S) \in \mathbb{R}^{(N \times N)}, O = PV \in \mathbb{R}^{(N \times d)}
$$

where the attention is applied row-wise.