import numpy as np
import moduloALC as alc
from collections.abc import Iterable


def calcularAx(A, x):
    """
    Calcula el producto matriz-vector con los parametros A y x
    """
    x = np.array(x).flatten()

    res = np.zeros(A.shape[0])  
    
    for i, row in enumerate(A):
        for j, value in enumerate(row):
            res[i] += value * x[j]
    return np.array(res).reshape((-1,1))

def normaInf(A):
    """
    Calcula la norma infinito de la matriz A
    """

    sumatorias = []
    for i, row in enumerate(A):
        sumatorias.append(sum(abs(row)))
    
    return max(sumatorias)

def esSimetrica(A, tol = 1e-8): 
    """
    Devuelve True si la matriz A es simetrica bajo la tolerancia 'tol'
    """

    for i, row in enumerate(A):
        for j in range(len(row)):
            if alc.error(A[i][j], A[j][i]) > tol:
                return False
            
    return True



def productoExterno(u, v):
    """
    Calcula el producto de u^t con v 
    """

    n = u.shape[0]
    res = np.zeros((n, n))
    for i, ui in enumerate(u):
        if ui[0] != 0:
            for j, vj in enumerate(v):
                res[i][j] = ui[0] * vj
    return res


def matricesIguales(A, B, atol = 1e-8):
    """
    Devuelve True si las matrices A y B son iguales indice a indice bajo la tolerancia 'atol'
    """

    if A.size != B.size and A[0].size != B[0].size:
        return False
    for i, fila in enumerate(A):
        for j, valor in enumerate(fila):
           if alc.error(np.float64(valor), np.float64(B[i][j])) > atol:
                return False
    return True


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

def traspuesta(A):
    """
    Calcula la matriz traspuesta de A 
    """

    if (len(A) == 0):
        return A

    elif (isinstance(A[0], Iterable)):
        A = np.array(A)
        m, n = A.shape
        res = np.zeros((n, m))
        for i, row in enumerate(A):
            for j, value in enumerate(row):
                res[j][i] = A[i][j]


    else:
        res = np.zeros((len(A),1))
        for i, value in enumerate(A):
            res[i][0] = value 
    
    return res

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

def productoEscalar(A, k):
    """
    Devuelve el producto escalar de la matriz A con la constante k 
    """
    return np.array(A) * k


def restaMatricial(A, B):
    """
    Devuelve el resultado de la resta matricial entre A y B
    """
    res = A.copy()
    for i in range(len(A)):
        res[i] = restaVectorial(A[i],B[i])
    return res

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

def restaVectorial(u, v):
    """
    Devuelve la resta vectorial entre u y v
    """

    return u - v

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