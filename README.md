# Linear Classifier from scratch

This project implements a multi-class linear classifier based on algebraic decompositions. It solves the fundamental equation $Wx = y$ to find the optimal weight matrix $W$ that maps input features to categorical targets.

## Mathematical Objective

The goal is to find the weight matrix $W \in \mathbb{R}^{m \times n}$ that minimizes the Mean Squared Error between our predictions ($Wx^{\mu}$) and the desired targets ($y^{\mu}$) across all $p$ training samples:

$$E(W) = \frac{1}{p} \sum_{\mu=1}^{p} \|y^{\mu} - Wx^{\mu}\|_2^2$$

In this equation:
* **$p$** is the total number of training patterns (e.g., 3000 images).
* **$x^{\mu}$** is the input embedding for sample $\mu$.
* **$y^{\mu}$** is the target vector in $\mathbb{R}^{2}$ representing the ground truth class.

We minimize this error by calculating the **Moore-Penrose Pseudoinverse** ($X^+$) to solve the system $W = YX^+$, where **$X$** is the feature matrix (stacking all $x^{\mu}$ as columns) and **$Y$** is the target matrix containing the one-hot encoded labels.

## Implemented Algorithms

The system provides three methods to calculate the weights $W$ from scratch:

### 1. Cholesky Factorization (Normal Equations)
Solves for the pseudoinverse by decomposing a symmetric positive-definite matrix into $LL^T$. It handles cases based on the dimensions of $X \in \mathbb{R}^{n \times p}$:
* **Case A**: If $rank(X)=p$ and $n>p$, then $X^+ = (X^T X)^{-1} X^T$.
* **Case B**: If $rank(X)=n$ and $n<p$, then $X^+ = X^T (X X^T)^{-1}$.

### 2. SVD (Singular Value Decomposition)
Finds the solution using the decomposition $X = U \Sigma V^T$. Here, **$U$** and **$V$** are orthogonal matrices representing the singular vectors, and **$\Sigma$** is a diagonal matrix containing the singular values. The weights are calculated as:
$$W = YV \Sigma^+ U^T$$

### 3. QR Decomposition
Calculates the decomposition of the transpose $X^T = QR$ using **Householder reflections**. In this method, **$Q$** is an orthogonal matrix and **$R$** is an upper triangular matrix. It solves the system $VR^T = Q$ (where $V = X^+$) to obtain the final weight matrix $W = YV$.

## Workflow

1. **Data Loading**: `load_dataset` reads the embeddings and creates the $X$ and $Y$ matrices.
2. **Training**: Computes $W$ using the selected algebraic method and saves it as a compressed file.
3. **Prediction**: Assigns a class to new inputs by finding the $argmax$ of the resulting prediction vector $y$, identifying the index with the highest probability.
