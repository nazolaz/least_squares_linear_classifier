from linalg import *
import numpy as np

def fit(X, Y, method='HH', tol=1e-10):
    match method:
        case 'HH' | 'GS':
            return fit_qr(X, Y, method, tol)
        case 'Cholesky':
            return fit_cholesky(X, Y)
        case 'SVD':
            return fit_svd(X, Y, tol)

def fit_cholesky(X, Y):
    """
    Returns the weight matrix W using the L matrix from Cholesky decomposition and Y, the target matrix.
    """

    n, p = X.shape
    rank_X = min(n, p)

    if rank_X == p and rank_X < n:
        L = compute_cholesky(X.T @ X)
        U_transpose = np.zeros((n, p))
        
        for i in range(n):
            y_i = forward_substitution(L, X[i])
            u_i = backward_substitution(L.T, y_i)
            U_transpose[i] = u_i

        W = Y @ (U_transpose.T)


    elif rank_X == n and rank_X < p:
        L = compute_cholesky(X @ X.T)

        V = np.zeros((p, n))

        for i in range(p):
            y_i = forward_substitution(L, X.T[i])
            V[i] = backward_substitution(L.T, y_i)

        W = Y @ V


    elif rank_X == p and p == n:
        X_inv = invert_LU(X)
        W = Y @ X_inv

    return W


def fit_svd(X, Y, tol=1e-10):
    """
    Returns the weight matrix W using the U (unitary) and S (diagonal) matrices from SVD decomposition and Y, the target matrix.
    """

    Ur, Sr, Vr = reduced_SVD(X, tol=tol)

    S_inv_diag = np.zeros((len(Sr), len(Sr)))
    for i in range(len(Sr)):
        S_inv_diag[i, i] = 1.0 / Sr[i]
    
    X_plus = Vr @ (S_inv_diag @ Ur.T)
    
    return Y @ X_plus


def fit_qr(X, Y, qr_method='HH', tol=1e-10):
    """
    Returns the weight matrix W using the QR decomposition matrices and Y, the target matrix.
    """

    # We solve for V by doing R * V.T = Q.T
    Q, R = compute_qr(X.T, qr_method, tol)

    m_r, n_r = R.shape
    m_p, n_p = Q.shape

    V = np.zeros((m_p, n_r))
 
    for i in range(m_p):
        b = Q[i] # this is equivalent to getting the Column of the transpose of Q
        V[i] = backward_substitution(R, b)

    return Y @ V