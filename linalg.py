import numpy as np

# ==========================================
# 1. AUXILIARES
# ==========================================

def filaCanonica(dimension, i):
    """
    Devuelve el vector canonico 'i' del espacio vectorial respectivo de 'dimension' en forma de vector fila
    """
    fila = np.zeros(dimension)
    fila[i] = 1
    return fila

def colCanonica(dimension, i):
    """
    Devuelve el vector canonico 'i' del espacio vectorial respectivo de 'dimension' en forma de vector columna
    """
    columna = np.zeros((dimension, 1))
    columna[i][0] = 1
    return columna

def norma(Xs, p):
    """
    Calcula la norma vectorial p de un vector Xs.
    
    Parámetros:
    Xs: Vector de numpy o lista de números.
    p: Orden de la norma (int) o la cadena 'inf' para norma infinito.
    
    Retorna el valor escalar de la norma.
    """
    if p == 'inf':
        return max(map(abs ,Xs))
    
    res = np.sum(np.abs(Xs) ** p)
    return res**(1/p)

def normalizarVector(vector, p):
    """
    Devuelve el vector pasado como parametro normalizado en norma p
    """
    normaVector = norma(vector, p)
    
    if normaVector == 0:
        return vector 
    
    return np.array(vector) / normaVector

def normaliza(Xs, p):
    """
    Normaliza una lista de vectores según la norma p indicada.
    
    Parámetros:
    Xs: Lista de vectores.
    p: Orden de la norma a utilizar.
    
    Retorna lista de vectores unitarios.
    """
    XsNormalizado = []

    for vector in Xs:
        res = normalizarVector(vector, p)
        XsNormalizado.append(res)

    return XsNormalizado

def signo(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0

def calcularAx(A, x):
    """
    Calcula el producto matriz-vector con los parametros A y x
    """
    x_flat = np.asarray(x).flatten()
    res = (A @ x_flat).astype(float)
    
    return res.reshape(-1, 1)

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

def extenderConIdentidad(A, p): #solo para matrices cuadradas
    """
    Devuelve la matriz A extendida hacia arriba a la izquierda con 1s en la diagonal
    """
    n = A.shape[0]
    res = np.eye(p)
    res[p-n:, p-n:] = A
    return res

def submatriz(A, l, k):
    """
    Recorta la matriz A desde A[l][l] hasta A[k][k]
    """
    return A[l-1:k, l-1:k]

def conseguirColumnaSufijo(A, j, k):
    """
    Extrae la columna j de la matriz A, pero solo considerando de los indices k para adelante
    """
    return A[k:A.shape[1], j]

def esSimetrica(A, tol = 1e-10): 
    """
    Devuelve True si la matriz A es simetrica bajo la tolerancia 'tol'
    """
    return np.allclose(A, A.T, tol)

# ==========================================
# 3. RESOLVEDORES DE SISTEMAS TRIANGULARES
# ==========================================

def sustitucionHaciaAtras(A, b):
    """
    Resuelve el sistema lineal Ax = b donde A es triangular superior.
    Retorna el vector solución x.
    """
    m, n = A.shape
    valoresX = np.zeros(n)

    for i in range(min(m, n) - 1, -1, -1):
        cocienteActual = A[i][i]

        if cocienteActual == 0:
            valoresX[i] = np.nan
        else:
            sumatoria = np.dot(A[i, i+1:], valoresX[i+1:])
            valoresX[i] = (b[i] - sumatoria) / cocienteActual

    return valoresX

def sustitucionHaciaDelante(A, b):
    """
    Resuelve el sistema lineal Ax = b donde A es triangular inferior.
    Retorna el vector solución x.
    """
    m, n = A.shape
    valoresX = np.zeros(n)

    for i in range(min(m, n)):
        cocienteActual = A[i][i]
        
        if cocienteActual == 0:
            valoresX[i] = np.nan
        else:
            sumatoria = np.dot(A[i, :i], valoresX[:i])
            valoresX[i] = (b[i] - sumatoria) / cocienteActual
    return valoresX

def res_tri(L, b, inferior=True):
    """
    Wrapper para resolver sistemas triangulares.
    """
    if(inferior):
        return sustitucionHaciaDelante(L,b)
    return sustitucionHaciaAtras(L,b)

# ==========================================
# 3. CHOLESKY Y LU
# ==========================================

def calculaLU(A):
    """
    Calcula la descomposición LU de la matriz A sin pivoteo.
    """
    m, n = A.shape
    Ac = A.copy()
    
    if m!=n:
        return None, None, 0

    for k in range(0, n-1):
        if Ac[k][k] == 0:
            return None, None, 0
        
        for i in range(k + 1, m):
            quotient = Ac[i][k]/Ac[k][k]    # Ac[i][k] es el iesimo elemento debajo del pivote que luego pasa a ser 0
            Ac[i][k] = quotient             # se guarda en Ac[i][k] el cociente para la eliminacion de gauss
            Ac[i, k+1 : n] = Ac[i, k+1 : n] - quotient * Ac[k, k+1 : n]

    return triangL(Ac), triangSup(Ac)

def inversaLU(A):
    """
    Calcula la inversa de A usando descomposición LU.
    """
    n = A.shape[0]

    L,U,_ = calculaLU(A)

    if (L is None or U is None):
        return None    

    Linv = np.zeros((n,n))
    Uinv = np.zeros((n,n))

    for i in range(n):
        colInv = res_tri(L, filaCanonica(n, i), inferior=True)
        for j in range(n):
            Linv[j][i] = colInv[j]

    for i in range(n):
        if( U[i,i] == 0):
            return None

        colInv = res_tri(U, filaCanonica(n, i), inferior=False)
        for j in range(n):
            Uinv[j][i] = colInv[j]

    return Uinv@Linv

def calculaLDV(A):
    """
    Calcula la descomposición L D V^T.
    """
    L, U = calculaLU(A)

    if(U is None):
        return None, None, None, 0

    Vt, D = calculaLU(U.T)

    if Vt is None:
        return None, None, None, 0
    
    return L, D, Vt.T

def esSDP(A, atol=1e-10):
    """
    Determina si la matriz A es Simétrica Definida Positiva (SDP).
    """
    if(not (esSimetrica(A, atol))):
        return False
    
    L, D, Lt = calculaLDV(A)

    if( D is None):
        return False
    
    for i in range(len(D)):
        if (D[i,i] <= 0):
            return False
    return True

def calculaCholesky(A):
    """
    Calcula la matriz L de la descomposición de Cholesky
    """
    if not esSDP(A):
        return None

    L, D, _ = calculaLDV(A)

    for i in range(len(D)): 
        D[i][i] = np.sqrt(D[i][i]) 

    return L@D


# ==========================================
# 4. ORTOGONALIZACION Y QR
# ==========================================

def QR_con_GS(A, tol=1e-10):
    """
    Calcula la factorización QR mediante el proceso de Gram-Schmidt.
    """
    m , n = A.shape
    Q = np.zeros((m,n))
    R = np.zeros((n,n))

    a_1 = A[:, 0]
    norma1 = norma(a_1, 2)
    R[0][0] = norma1

    if norma1 > tol:
        Q[:, 0] = normalizarVector(a_1, 2)
    else:
        Q[:, 0] = a_1

    for j in range(1, n):
        qMoño_j = A[:, j]

        for k in range(0, j):
            q_k = Q[:, k]
            R[k][j] = np.vdot(q_k, qMoño_j)
            qMoño_j = qMoño_j - (q_k * R[k][j])
        
        R[j][j] = norma(qMoño_j, 2)

        if R[j][j] > tol:

            Q[:, j] = qMoño_j * 1/R[j][j]
        else:
            Q[:, j] = qMoño_j

    return Q, R

def QR_con_HH(A, tol=1e-10):
    """
    Calcula la factorización QR mediante reflexiones de Householder.
    """
    # OPTIMIZACIÓN
    # H = I - 2 * vv^t
    # H A = (I - 2 * vv^t) A
    # H A = A - 2 * v (v^tA)

    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)

    if m < n:
        return None, None

    for k in range(min(m,n)):
        
        # x es el vector columna actual desde la diagonal hacia abajo
        x = R[k:, k]
        
        norm_x = norma(x, 2)
        if norm_x < tol:
            continue
            
        signo_x = signo(x[0])
        # u = x + signo_x * ||x|| * e1
        u = x.copy()
        u[0] += signo_x * norm_x
        
        v = u / norma(u, 2)
        v_fila = v.reshape(1, -1)

        valor_intermedio = (v_fila@R[k:, k:]).flatten()
        R[k:, k:] -= 2 * np.outer(v, valor_intermedio)
        v_columna = v.reshape(-1, 1)
        
        # Qv (m, n-k) x (n-k, n-k)
        valor_intermedio_Q = (Q[:, k:]@v_columna).flatten()
        
        # Q = Q - 2 Qv * v^t
        Q[:, k:] -= 2 * np.outer(valor_intermedio_Q, v)

    return Q, R

def calculaQR(A, metodo='HH', tol=1e-10):
    """
    Calcula la descomposición QR de A.
    """
    if metodo == 'HH':
        return QR_con_HH(A, tol)
    
    elif metodo == 'GS':
        return QR_con_GS(A, tol)
    
    else: 
        return None, None, None


# ==========================================
# 5. SVD Y DIAGONALIZACION
# ==========================================
def diagRH(A, tol=1e-10, K=100, iteracion = 0):
    n = len(A)

    v1, l1, _ = metpot2k(A, tol, K)
    u = normalizarVector((colCanonica(n,0) - v1),2).flatten()
    Au = A @ u
    uAu = np.dot(u, Au)
    q = Au - uAu * u
    W = np.outer(q, u)
    
    mid = A - 2 * (W + W.T)

    if n == 2:
        uut = np.outer(u, u)
        Anew = np.eye(n) - 2 * uut
        return Anew, mid
    
    Amoño = submatriz(mid, 2, n)
    Smoño, Dmoño = diagRH(Amoño, tol, K, iteracion + 1)

    D = extenderConIdentidad(Dmoño, n)
    D[0][0] = l1

    Smoño_ext = extenderConIdentidad(Smoño, n)
    S = Smoño_ext - 2 * np.outer(u, u @ Smoño_ext)

    return S, D

def vectorValoresSingulares(SigmaHat, k):
    """
    Extrae y calcula los valores singulares a partir de la matriz diagonal de autovalores.

    Calcula la raíz cuadrada de los primeros k elementos diagonales de SigmaHat (usualmente 
    provenientes de la diagonalización de A^T A) para obtener los valores singulares.

    Parámetros:
    SigmaHat: Matriz diagonal conteniendo los cuadrados de los valores singulares.
    k: Cantidad de valores a extraer.

    Retorna:
    Lista con los primeros k valores singulares.
    """

    SigmaHatVector = list()
    for i in range(k):
            SigmaHatVector.append(np.sqrt(np.abs(SigmaHat[i][i])))
    return SigmaHatVector

def svd_reducida(A, k="max", tol=1e-10):
    """
    Calcula la Descomposición en Valores Singulares (SVD) reducida de A.
    """
    m, n = A.shape

    # chequeo de dimension para optimizar
    usar_traspuesta = False
    if m < n:
        A = A.T
        usar_traspuesta = True

    m, n = A.shape

    AtA = (A.T)@ A
    VHat_full, SigmaHat = diagRH(AtA, tol=tol, K=100)

    # calculo de rango
    rango=min(m, n)
    for i in range(len(SigmaHat)):
        if SigmaHat[i,i] < tol:
            rango = i
            break
    rango = min(m, n, rango)
    k = rango if k == "max" else k

    # tomamos las primeras k columnas de Vhat y los primeros k valores singulares
    VHat_k = VHat_full[:, :k]
    SigmaHatVector = vectorValoresSingulares(SigmaHat, k)

    B = A@VHat_k
    UHat_k = B
    for j in range(k): 
        sigma = SigmaHatVector[j]
        UHat_k[:, j] /= sigma

    if usar_traspuesta:
        return VHat_k, SigmaHatVector, UHat_k
    else:
        return UHat_k, SigmaHatVector, VHat_k

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

def f_A(A, v):
    wprima = calcularAx(A, v)

    if norma(wprima, 2) > 0:
        return normalizarVector(wprima, 2) 

    return 0

def metpot2k(A, tol=1e-10, K=100.0):
    """
    Calcula el autovalor dominante y su autovector asociado usando el método de la potencia.
    """
    n = len(A[0])
    v = np.random.rand(n,1)
    vmoñotemp = f_A(A, v)
    vmoño = f_A(A, vmoñotemp)
    e = float(np.vdot(vmoño, v))
    k = 0
    while(1 - abs(e) > tol and k < K):
        v = vmoño
        vmoño = f_A(A, vmoño)
        e = float(np.vdot(vmoño, v))
        k = k + 1
    
    ax = calcularAx(A, vmoño)
    autovalor = np.vdot(vmoño, ax)

    return vmoño, autovalor, k