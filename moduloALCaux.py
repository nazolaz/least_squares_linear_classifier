import numpy as np
import moduloALC as alc
from collections.abc import Iterable


def calcularAx(A, x):
    """
    Calcula el producto matriz-vector con los parametros A y x
    """
    x_flat = np.asarray(x).flatten()
    res = (A @ x_flat).astype(float)
    
    return res.reshape(-1, 1)


def esSimetrica(A, tol = 1e-8): 
    """
    Devuelve True si la matriz A es simetrica bajo la tolerancia 'tol'
    """
            
    return np.allclose(A, A.T, tol)


def filaCanonica(dimension, i):
    """
    Devuelve el vector canonico 'i' del espacio vectorial respectivo de 'dimension' en forma de vector fila
    """
    fila = np.zeros(dimension)
    fila[i] = 1
    return fila

def colCanonico(dimension, i):
    """
    Devuelve el vector canonico 'i' del espacio vectorial respectivo de 'dimension' en forma de vector columna
    """

    columna = np.zeros((dimension, 1))
    columna[i][0] = 1
    return columna

def normalizarVector(vector, p):
    """
    Devuelve el vector pasado como parametro normalizado en norma p
    """

    normaVector = alc.norma(vector, p)
    
    if normaVector == 0:
        return vector 
    
    return np.array(vector) / normaVector


def triangSup(A):
    """
    Devuelve la matriz A pero con 0s debajo de la diagonal
    """
    ATriangSup = A.copy()

    for i in range(len(A)):
        for j in range(len(A[i])):
            if j < i:
                ATriangSup[i,j] = 0
    
    return ATriangSup

def triangL(A):
    """
    Devuelve la matriz A pero con 0s sobre la diagonal y reescribiendo la diagonal con 1s
    """
    L = A.copy()

    for i in range(len(A)):
        for j in range(len(A[i])):
            if j > i:
                L[i][j] = 0 
            if j == i:
                L[i][i] = 1
    
    return L


def conseguirColumnaSufijo(A, j, k):
    """
    Extrae la columna j de la matriz A, pero solo considerando de los indices k para adelante
    """
        
    return A[k:A.shape[1], j]

def productoInterno(u, v):
    """
    Calcula el producto u * v^t
    """    
    
    subtotal = 0
    
    for ui, vi in zip(u.flat, v.flat):
        subtotal += ui * vi
    
    return subtotal


def extenderConIdentidad(A, p): #solo para matrices cuadradas
    """
    Devuelve la matriz A extendida hacia arriba a la izquierda con 1s en la diagonal
    """
    res = nIdentidad(p)
    n = A.shape[0]
    for i in range(p - n, p):
        k = i - (p - n)
        for j in range(p - n, p):
            l = j - (p - n)
            res[i][j] = A[k][l]
    return res


def nIdentidad(n):
    """
    Devuelve la matriz identidad de R^n
    """
    I = np.zeros((n,n))
    for k in range(n):
        I[k][k] = 1
    return I

def signo(n):
    
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0
    
def submatriz(A, l, k):
    """
    Recorta la matriz A desde A[l][l] hasta A[k][k]
    """
    return A[l-1:k, l-1:k]

def calculaCholesky(A):
    """
    Calcula la matriz L de la descomposición de Cholesky
    """
    
    if not alc.esSDP(A):
        return None

    L, D, _, _ = alc.calculaLDV(A)

    for i in range(len(D)): # type: ignore
        D[i][i] = np.sqrt(D[i][i]) # type: ignore

    return L@D


def reducirSVD(U, S, V):
    """
    Calcula la version reducida de la SVD a partir de la version completa
    """
    # Convertir listas a numpy arrays
    U = np.array(U) if isinstance(U, list) else U
    V = np.array(V) if isinstance(V, list) else V
    S = np.array(S) if isinstance(S, list) else S
    m, n = U.shape[0], V.shape[1]
    rango = min(m,n)

    Slist = list()
    if len(S.shape) == 1: # Si S es lista de valores singulares
        Slist = S[:rango]
    elif len(S.shape) == 2: # Si S es matriz diagonal con valores singulares
            Slist = [S[i, i] for i in range(rango)]
    else:
        raise TypeError(f'Argumento S={S} inválido: debe ser una lista de valores singulares o una matriz diagonal de valores singulares. ')

    U_red = U[:, :rango]
    V_red = V[:, :rango]

    return U_red, Slist, V_red