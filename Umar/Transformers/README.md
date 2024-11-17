# Transformer from Scartch

Following the [video from Umar](https://www.youtube.com/watch?v=ISNdQcPhsts&t=229s)

> [!NOTE] 
> Badly explained for the moment

## High Level Exaplanation

The high level input of a transformer is a Input of shape of $x \in \mathbb{R}$ of $dim \in (seq,d_{input})$. This input is separated into 3 different matrices called **Query, Key and Value**. of shape $Q \in \mathbb{R}$ of $dim \in (seq,d_{input})$, $K \in \mathbb{R}$ of $dim \in (seq,d_{input})$ and $V \in \mathbb{R}$ of $dim \in (seq,d_{input})$ respectively.

These then are multiplied by the weight matrices $W^Q \in \mathbb{R}$ of $dim \in (d_{input},d_{input})$, $W^K \in \mathbb{R}$ of $dim \in (d_{input},d_{input})$ and $W^V \in \mathbb{R}$ of $dim \in (d_{input},d_{input})$ respectively. using the following equations

$$
\normalsize Q' = XW^Q    \tiny (seq,d_{input}) \times (d_{input},d_{input}) = (seq,d_{input}) \\
\normalsize K' = XW^K    \tiny  (seq,d_{input}) \times (d_{input},d_{input}) = (seq,d_{input}) \\
\normalsize V' = XW^V    \tiny   (seq,d_{input}) \times (d_{input},d_{input}) = (seq,d_{input}) \\
$$

These multiplied matrices are then split into a number equal to the number of $h$ heads. So the $Q'$, $K'$ and $V'$ are split into $Q_1, Q_2, Q_3, ... Q_h$, $K_1, K_2, K_3, ... K_h$ and $V_1, V_2, V_3, ... V_h$ respectively of shape $Q_i | \mathbb{I} \in (1,...,h) $ of $dim \in (seq,d_{input}/h)$, $K_i | \mathbb{I} \in (1,...,h) $ of $dim \in (seq,d_{input}/h)$ and $V_i | \mathbb{I} \in (1,...,h) $ of $dim \in (seq,d_{input}/h)$ respectively.

Let's first define the softmax function as: 

$$
\normalsize softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

This softmax function is applied to the dot product of $Q_i$ and $K_i$ and $V_i$ to get the attention scores of the 1sty head via the following equation: Note $d_{k} = d_{input}/h$

$$
\normalsize Attention(Q_i,K_i,V_i) = softmax(\frac{Q_iK_i^T}{\sqrt{d_{input}/h}})V_i
$$

with an output of $Attention(Q_i,K_i,V_i) \in \mathbb{R}$ of $dim \in (seq,d_{input}/h)$. All of thse outputs are then concatenated into an output $H \in \mathbb{R}$ of $dim \in (seq,h * d_{k})$ and then multiplied by the weight matrix $W^O \in \mathbb{R}$ of $dim \in (d_k * h,d_{input})$ to get the final output of the transformer. 

So the operation on each head is as follows:

$$
head_i = Attention(Q \cdot W^Q_i, K \cdot W^K_i, V \cdot W^V_i) \tiny (seq,d_{input}) \times (d_{input},d_{input}) = (seq,d_{input})
$$ 

where all the concatenated heads are then multiplied by the weight matrix $W^O$ to get the final output of the transformer.

$$
MultiHead(Q,K,V) = Concat(head_1,head_2,...,head_h) \cdot W_O \tiny   (seq,h * d_{k}) \times (d_k * h,d_{input}) = (seq,d_{input})
$$

Andn the Final output of the transformer is:

$$
\normalsize Output = HW^O    \tiny (seq,h * d_{k}) \times (d_k * h,d_{input}) = (seq,d_{input})
$$