import numpy as np
from moduloALCaux import *


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


def sustitucionHaciaAtras(A, b):
    """
    Resuelve el sistema lineal Ax = b donde A es triangular superior.
    
    Retorna el vector solución x.
    """
    m, n = A.shape
    valoresX = np.zeros(n)

    for i in range( min(A.shape) -1, -1, -1):
        cocienteActual = A[i][i]
        sumatoria = 0
        for k in range(i + 1, n):
            sumatoria += A[i][k] * valoresX[k]

        if cocienteActual == 0:
            valoresX[i] = np.nan  
        else:
            valoresX[i] = (b[i] - sumatoria) / cocienteActual
    return valoresX

def sustitucionHaciaDelante(A, b):
    """
    Resuelve el sistema lineal Ax = b donde A es triangular inferior.
    
    Retorna el vector solución x.
    """
    valoresX = []
    for i, row in enumerate(A):
        cocienteActual = row[i]
        sumatoria = 0
        for k in range(i):
            sumatoria += A[i][k] * valoresX[k]
        valoresX.append((b[i] - sumatoria)/cocienteActual)
    return np.array(valoresX)

def res_tri(L, b, inferior=True):
    """
    Wrapper para resolver sistemas triangulares.
    
    Parámetros:
    inferior: Si es True, asume L triangular inferior
              Si es False, asume L triangular superior

    Retorna el vector solución x.
    """
    if(inferior):
        return sustitucionHaciaDelante(L,b)
    return sustitucionHaciaAtras(L,b)

def calculaLU(A):
    """
    Calcula la descomposición LU de la matriz A sin pivoteo.
    
    Retorna:
    L: Matriz triangular inferior con 1s en la diagonal.
    U: Matriz triangular superior.
    cant_op: Número de operaciones de punto flotante realizadas.
    
    Devuelve (None, None, 0) si encuentra un 0 en la diagonal (necesita pivoteo).
    """
    cant_op = 0
    m, n = A.shape
    Ac = A.copy()
    
    if m!=n:
        return None, None, 0

    for k in range(0, n-1):
        if A[k][k] == 0:
            return None, None, 0
        
        for i in range(k + 1, n):
            
            mi = Ac[i][k]/Ac[k][k]
            cant_op += 1
            Ac[i][k] = mi
            for j in range(k+1, m):
                Ac[i][j] = Ac[i][j] - mi * Ac[k][j]
                cant_op += 2 
    
    return triangL(Ac), triangSup(Ac), cant_op

def inversa(A):
    """
    Calcula la inversa de A usando descomposición LU.
    
    Retorna:
    La matriz inversa A^-1 o None si A es singular o requiere pivoteo no soportado.
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
    
    Descompone A en L * U, y luego descompone U^T para extraer la diagonal D.
    
    Retorna:
    L: Triangular inferior unitaria.
    D: Matriz diagonal.
    Vt: Matriz V traspuesta (triangular superior unitaria).
    nops: Cantidad total de operaciones.
    """
    L, U, nops1 = calculaLU(A)

    if(U is None):
        return None, None, None, 0

    Vt, D, nops2 = calculaLU(U.T)


    if Vt is None:
        return None, None, None, 0
    
    return L, D, Vt.T, nops1 + nops2

def esSDP(A, atol=1e-10):
    """
    Determina si la matriz A es Simétrica Definida Positiva (SDP).
    
    Verifica simetría y luego chequea que todos los elementos de la matriz D 
    en la descomposición LDL^T sean estrictamente positivos.
    """
    if(not (esSimetrica(A, atol))):
        return False
    
    L, D, Lt, _ = calculaLDV(A)

    if( D is None):
        return False
    
    for i in range(len(D)):
        if (D[i,i] <= 0):
            return False
    return True


def metpot2k(A, tol=1e-15, K=1000.0):
    """
    Calcula el autovalor dominante y su autovector asociado usando el Método de la Potencia.
    
    Parámetros:
    A: Matriz cuadrada.
    tol: Tolerancia para el criterio de parada.
    K: Número máximo de iteraciones.
    
    Retorna:
    v: Autovector normalizado asociado al autovalor dominante.
    l: Autovalor dominante estimado.
    k: Número de iteraciones realizadas.
    """
    n = len(A[0])
    v = np.random.rand(n,1)
    vmoñotemp = f_A(A, v)
    vmoño = f_A(A, vmoñotemp)
    e = float(productoInterno(vmoño, v))
    k = 0
    while( abs(e - 1) > tol and k < K):

        v = vmoño
        vmoñotemp = f_A(A, v)
        vmoño = f_A(A, vmoñotemp)
        e = float(productoInterno(vmoño, v))
        k = k + 1
    
    ax = calcularAx(A, vmoño)
    autovalor = productoInterno(vmoño, ax)

    return vmoño, autovalor, k


def f_A(A, v):

    wprima = calcularAx(A, v)

    if norma(wprima, 2) > 0:
        return normalizarVector(wprima, 2) 


    return 0

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    Calcula la factorización QR mediante el proceso de Gram-Schmidt.
    
    Retorna:
    Q: Matriz ortogonal.
    R: Matriz triangular superior.
    nops (opcional): Cantidad de operaciones realizadas.
    """
    m , n = A.shape
    Q = np.zeros((m,n))
    R = np.zeros((n,n))
    nops = 0

    a_1 = A[:, 0]
    norma1 = norma(a_1, 2)
    R[0][0] = norma1
    nops += 2*m - 1

    if norma1 > tol:
        Q[:, 0] = normalizarVector(a_1, 2)
    else:
        Q[:, 0] = a_1

    for j in range(1, n):
        qMoño_j = A[:, j]

        for k in range(0, j):
            q_k = Q[:, k]
            R[k][j] = productoInterno(q_k, qMoño_j)
            nops += 2*m- 1
            qMoño_j = qMoño_j - (q_k * R[k][j])
            nops += 2*m
        
        R[j][j] = norma(qMoño_j, 2)
        nops += 2*m - 1

        if R[j][j] > tol:

            Q[:, j] = qMoño_j * 1/R[j][j]
            nops += 1
        else:
            Q[:, j] = qMoño_j

    if (retorna_nops):
        return Q, R, nops

    return Q, R

def QR_con_HH(A, tol=1e-12):
    """
    Calcula la factorización QR mediante reflexiones de Householder.
    
    Retorna:
    Q: Matriz ortogonal.
    R: Matriz triangular superior.
    """

    # OPTIMIZACIÓN
    # H = I - 2 * vv^t
    # H A = (I - 2 * vv^t) A
    # H A = A - 2 * v (v^tA)

    m, n = A.shape
    
    R = A.copy()
    Q = nIdentidad(m)

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

def calculaQR(A, metodo='RH', tol=1e-12, nops=False):
    """
    Permite calcular la descomposición QR de A.
    
    Parámetros:
    metodo: 'RH' (Reflexiones Householder) o 'GS' (Gram-Schmidt).
    
    Retorna Q y R.
    """
    if metodo == 'RH':
        return QR_con_HH(A, tol)
    
    elif metodo == 'GS':
        if nops:
            return QR_con_GS(A, tol, True)
        else:
            return QR_con_GS(A, tol)
    
    else: 
        return None, None, None
    
def diagRH(A, tol=1e-15, K=1000):
    """
    Calcula autovalores y autovectores de una matriz simétrica mediante deflación de Householder.
    
    Retorna:
    S: Matriz cuyas columnas son los autovectores.
    D: Matriz diagonal de autovalores.
    """
    n = len(A)
    v1, l1, _ = metpot2k(A, tol, K)
    resta = normalizarVector((colCanonico(n,0) - v1),2)
    producto = np.outer(resta, resta)  
    Hv1 = nIdentidad(n) - (producto * 2)
    mid = Hv1@(A@(Hv1.T))

    if n == 2:
        return Hv1, mid
    
    Amoño = submatriz(mid, 2, n)
    Smoño, Dmoño = diagRH(Amoño, tol, K)

    D = extenderConIdentidad(Dmoño, n)
    D[0][0] = l1

    S = Hv1@extenderConIdentidad(Smoño, n)

    return S, D


def svd_reducida(A, k="max", tol=1e-15):
    """
    Calcula la Descomposición en Valores Singulares (SVD) reducida de A.
    
    Utiliza la diagonalización de A^T A para obtener V y Sigma, y luego proyecta para obtener U.
    
    Parámetros:
    k: Número de valores singulares a retener (o "max" para rango completo detectado).
    tol: Tolerancia para considerar un valor singular como cero.
    
    Retorna:
    U_k: Primeras k columnas de la matriz unitaria izquierda.
    S_k: Vector con los primeros k valores singulares.
    V_k: Primeras k columnas de la matriz unitaria derecha (V, no V^T).
    """

    m, n = A.shape

    # chequeo de dimension para optimizar
    usar_traspuesta = False
    if m < n:
        A = A.T
        usar_traspuesta = True

    m, n = A.shape

    AtA = (A.T)@ A
    VHat_full, SigmaHat = diagRH(AtA, tol=tol, K=10000)

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
    for j in range(k): # type: ignore
        sigma = SigmaHatVector[j]
        for fila in range(m):
            UHat_k[fila][j] = UHat_k[fila][j] / sigma
    if usar_traspuesta:
        return VHat_k, SigmaHatVector, UHat_k
    else:
        return UHat_k, SigmaHatVector, VHat_k

def vectorValoresSingulares(SigmaHat, k):
    SigmaHatVector = list()
    for i in range(k):
            SigmaHatVector.append(np.sqrt(np.abs(SigmaHat[i][i])))
    return SigmaHatVector