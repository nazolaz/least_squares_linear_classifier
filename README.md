# Linear Classifier from Scratch

This project implements a generalized **Multi-Class Linear Classifier**. It takes input features and learns to categorize them by solving the fundamental linear equation:
$$Wx = y$$

## Mathematical Objective

The goal is to find the weight matrix $W$ that minimizes the **Mean Squared Error** between our predictions and the ground truth:
$$E(W) = \frac{1}{p} \sum_{\mu=1}^{p} \|y^{\mu} - Wx^{\mu}\|_2^2$$

We achieve this by calculating the **Moore-Penrose Pseudoinverse** ($X^+$):
$$W = YX^+$$



## Implemented Algorithms

I built three different solvers to calculate the weights $W$ from scratch:

### 1. SVD (Singular Value Decomposition)
This is the most stable method. It decomposes the matrix $X$ into $U \Sigma V^T$ to find the pseudoinverse:
$$X^+ = V \Sigma^+ U^T$$

It is particularly useful for handling redundant data or features that contain mostly zeros.



### 2. QR Decomposition
We solve for the weights by decomposing the transpose of the input matrix ($X^T = QR$). I implemented this using **Householder reflections**, solving the system:
$$VR^T = Q$$

This method is efficient for full-rank datasets and provides a stable alternative to the normal equations.

### 3. Cholesky Factorization
This solver uses the **Normal Equations**. It decomposes a symmetric positive-definite matrix into $LL^T$.
* **Case A**: If $rank(X)=p$ and $n>p$, then $X^+ = (X^T X)^{-1} X^T$.
* **Case B**: If $rank(X)=n$ and $n<p$, then $X^+ = X^T (X X^T)^{-1}$.


## Performance Benchmarking (MNIST)

To validate the model, I tested the SVD solver on the **MNIST dataset**. Below is the class distribution achieved during a test run:

| Digit | Distribution | Digit | Distribution |
| :--- | :--- | :--- | :--- |
| **0** | 10.15% | **5** | 7.36% |
| **1** | 13.34% | **6** | 10.24% |
| **2** | 8.69% | **7** | 10.67% |
| **3** | 10.77% | **8** | 8.52% |
| **4** | 10.59% | **9** | 9.67% |


## Workflow

* **Data Loading**: The system automatically detects classes and features using one-hot encoding for targets.
* **Training**: Calculates $W$ using the chosen algorithm and saves it as a compressed `.npz` file.
* **Prediction**: Multiplies new inputs by the trained $W$ and applies an $argmax$ to determine the final class.