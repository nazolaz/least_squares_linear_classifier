from moduloALC import *
from moduloALCaux import *
import numpy as np

def pinvEcuacionesNormales(X, L, Y):
    """
    Devuelve la matriz de pesos W utilizando la matriz L de la decomposicion de Cholesky e Y, la matriz de targets
    """

    n, p = X.shape
    rangoX = min(n, p)
    Lt = traspuesta(L)zz

    if rangoX == p and rangoX < n:

        Utraspuesta = np.zeros((n,p))
        
        for i in range(n):
        
            y_i = sustitucionHaciaDelante(L, X[i]) # iesima columna de X traspuesta
            u_i = sustitucionHaciaAtras(Lt, y_i)
            Utraspuesta[i] = u_i

        U = traspuesta(Utraspuesta)
        W = productoMatricial(Y, U)


    elif rangoX == n and rangoX < p:

        V = np.zeros((p,n))
        Xtraspuesta = traspuesta(X)

        for i in range(p):
            y_i = sustitucionHaciaDelante(L, Xtraspuesta[i]) # iesima columna de X
            V[i] = sustitucionHaciaAtras(Lt, y_i)

        W = productoMatricial(Y, V)


    elif rangoX == p and p == n:
        Xinv = inversa(X)
        W = productoMatricial(Y, Xinv)

    return W


def pinvSVD(U, S, V, Y):
    """
    Devuelve la matriz de pesos W utilizando las matrices U (unitaria) y S (diagonal) de la decomposicion SVD e Y, la matriz de targets
    """

    Ur, Sr, Vr = reducirSVD(U, S, V)

    S_inv_diag = np.zeros(Ur.shape)
    for i in range(len(Sr)):
        S_inv_diag[i,i] = 1.0 / Sr[i]
    
    X_plus = productoMatricial(Vr, productoMatricial(S_inv_diag, traspuesta(Ur)))
    
    W = productoMatricial(Y, X_plus)
    
    return W


def pinvHouseHolder(Q, R, Y):
    """
    Devuelve la matriz de pesos W pasando como parametros las matrices Q y R conseguidas utilizando HouseHolder e Y, la matriz de targets
    """
    return qrFCN(Q, R, Y)

def pinvGramSchmidt(Q, R, Y):
    """
    Devuelve la matriz de pesos W pasando como parametros las matrices Q y R conseguidas utilizando Gram-Schmidt e Y, la matriz de targets
    """
    return qrFCN(Q, R, Y)


def qrFCN(Q, R, Y):
    """
    Devuelve la matriz de pesos W utilizando las matrices de la descomposicion QR e Y, la matriz de targets
    """

    #despejamos V haciendo R * V.T = Q.T
    m_r, n_r = R.shape # shape R (2000, 1536)
    m_p, n_p = Q.shape # shape Q (2000, 2000)

    V = np.zeros((m_p, n_r)) # shape V.T (1536, 2000) -> shape V (2000, 1536) 
 
    
    for i in range(m_p):
        b = Q[i] # esto es equivalente a conseguirColumna(traspuesta(Q))
        V[i] = sustitucionHaciaAtras(R, b)

    return productoMatricial(Y,  V)


    
def esPseudoInversa(X, pX, tol= 1e-8):
    """
    Devuelve True si pX es la pseudoinversa de X bajo los criterios de Moore-Penrose y la tolerancia 'tol'
    """
    X_pX = productoMatricial(X, pX)
    pX_X = productoMatricial(pX, X)

    condicion1 = matricesIguales(X, productoMatricial(X,pX_X), tol)
    condicion2 = matricesIguales(pX, productoMatricial(pX_X,pX), tol)
    condicion3 = esSimetrica(X_pX, tol)
    condicion4 = esSimetrica(pX_X, tol)

    return condicion1 & condicion2 & condicion3 & condicion4


