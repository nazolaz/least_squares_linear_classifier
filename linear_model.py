from linalg import *
import numpy as np

def fit_cholesky(X, Y):
    """
    Devuelve la matriz de pesos W utilizando la matriz L de la decomposicion de Cholesky e Y, la matriz de targets
    """

    n, p = X.shape
    rangoX = min(n, p)

    if rangoX == p and rangoX < n:
        L = calculaCholesky(X.T @ X) 
        Utraspuesta = np.zeros((n,p))
        
        for i in range(n):
        
            y_i = sustitucionHaciaDelante(L, X[i]) 
            u_i = sustitucionHaciaAtras(L.T, y_i)
            Utraspuesta[i] = u_i

        W = Y @ (Utraspuesta.T)


    elif rangoX == n and rangoX < p:
        L = calculaCholesky(X @ X.T) 

        V = np.zeros((p,n))

        for i in range(p):
            y_i = sustitucionHaciaDelante(L, X.T[i]) 
            V[i] = sustitucionHaciaAtras(L.T, y_i)

        W = Y@V


    elif rangoX == p and p == n:
        Xinv = inversa(X)
        W = Y @ Xinv

    return W


def fit_svd(X, Y, tol = 1e-15):
    """
    Devuelve la matriz de pesos W utilizando las matrices U (unitaria) y S (diagonal) de la decomposicion SVD e Y, la matriz de targets
    """

    Ur, Sr, Vr = svd_reducida(X, tol)

    S_inv_diag = np.zeros((len(Sr), len(Sr)))
    for i in range(len(Sr)):
        S_inv_diag[i,i] = 1.0 / Sr[i]
    
    X_plus = Vr @ (S_inv_diag @ Ur.T)
    
    return Y @ X_plus


def fit_qr(X, Y, qr_method = 'RH', tol = 1e-15):
    """
    Devuelve la matriz de pesos W utilizando las matrices de la descomposicion QR e Y, la matriz de targets
    """

    #despejamos V haciendo R * V.T = Q.T
    Q, R = calculaQR(X, qr_method, tol)

    m_r, n_r = R.shape # shape R (2000, 1536)
    m_p, n_p = Q.shape # shape Q (2000, 2000)

    V = np.zeros((m_p, n_r)) # shape V.T (1536, 2000) -> shape V (2000, 1536) 
 
    
    for i in range(m_p):
        b = Q[i] # esto es equivalente a conseguirColumna(traspuesta(Q))
        V[i] = sustitucionHaciaAtras(R, b)

    return Y @ V
