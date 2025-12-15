import numpy as np
from moduloALCaux import *

def error(x, y):
    """
    Calcula el error absoluto entre dos escalares.

    Retorna La diferencia absoluta |x - y| como float64.
    """
    return abs(np.float64(x) - np.float64(y))

def error_relativo(x,y):
    """
    Calcula el error relativo entre un valor real x y un valor aproximado y.
    
    Si x es 0, devuelve el error absoluto de y para evitar división por cero.
    
    Retorna |x - y| / |x| (o |y| si x=0).
    """
    if x == 0:
        return abs(y)
    return error(x,y)/abs(x)

def sonIguales(x,y,atol=1e-08):
    """
    Determina si dos valores x e y son iguales bajo una tolerancia absoluta.
    
    Parámetros:
    atol: Tolerancia absoluta (por defecto 1e-08).
    
    Retorna True si |x - y| <= atol, False en caso contrario.
    """
    
    return np.allclose(error(x,y),0,atol=atol)

def rota(theta: float):
    """
    Genera una matriz de rotación 2x2 para un ángulo dado en el plano Euclídeo.
    
    Parámetros:
    theta: Ángulo de rotación en radianes (sentido antihorario).
    
    Retorna matriz de numpy 2x2 de rotación.
    """
    cos = np.cos(theta)
    sen = np.sin(theta)
    
    return np.array([[cos,-sen],[sen,cos]])

def escala(s):
    """
    Genera una matriz diagonal de escalado.
    
    Parámetros:
    s: Lista o vector con los factores de escala para cada dimensión.
    
    Retorna matriz diagonal donde M[i,i] = s[i].
    """
    matriz = np.eye(len(s))

    for i in range(len(s)):
        matriz[i][i] = s[i]

    return matriz

def rota_y_escala(theta: float, s):
    """
    Genera una matriz compuesta de escalado seguido de una rotación.
    
    El orden de aplicación es primero Escala, luego Rotación (R * S).
    
    Retorna matriz de numpy resultante de multiplicar R(theta) @ S(s).
    """
    return productoMatricial(escala(s), rota(theta))

def afin(theta, s, b):
    """
    Construye la matriz de transformación afín (homogénea 3x3) usando rotación theta, escala s y traslación b.

    Parámetros:
    theta: Ángulo de rotación.
    s: Factores de escala.
    b: Vector de traslación (tx, ty).
    """

    m1 = rota_y_escala(theta, s)
    return np.array([[m1[0][0],m1[0][1], b[0]],[m1[1][0], m1[1][1], b[1]],[0,0,1]])

def trans_afin(v, theta, s, b):
    """
    Aplica una transformación afín a un vector 2D v.
    
    Utiliza coordenadas homogéneas internamente para aplicar R, S y la traslación b.
    
    Retorna vector 2D transformado.
    """
    casi_res = productoMatricial(afin(theta, s, b),np.array([[v[0]],[v[1]],[1]]))
    return np.array([casi_res[0][0], casi_res[1][0]])

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

def normaExacta(A, p = [1, 'inf']):
    """
    Calcula la norma matricial inducida exacta (Norma-1 o Norma-Infinito).
    
    Parámetros:
    A: Matriz de numpy.
    p: 1 (máxima suma absoluta de columnas) o 'inf' (máxima suma absoluta de filas).
    
    Retorna el valor de la norma o None si p no es soportado.
    """

    if p == 1:
        # la norma 1 de A es igual a la norma infinito de A.T
        return normaInf(traspuesta(A))
    
    elif p == 'inf':
        return normaInf(A)
    
    else:
        return None

def normaMatMC(A, q, p, Np):
    """
    Estima la norma inducida matricial ||A||_{p,q} mediante el método de Monte Carlo.
    
    Genera Np vectores aleatorios unitarios en norma p, les aplica A, y mide su tamaño en norma q.
    
    Retorna una lista con: [maxima_norma_hallada, vector_que_la_alcanza].
    """
    n = A.shape[0]
    vectors = []

    ## generamos Np vectores random
    for _ in range(0,Np):
        vectors.append(np.random.rand(n,1)*2-1)
    
    ## normalizamos los vectores
    normalizados = normaliza(vectors, p)

    ## multiplicar A por cada Xs
    multiplicados = []
    for Xs in normalizados:
        multiplicados.append(calcularAx(A, Xs).flatten())
    
    maximo = [0,0] # (máxima norma, máximo vector)
    for vector in multiplicados:
        
        if norma(vector, q) > maximo[0]:
            maximo[0] = norma(vector, q)
            maximo[1] = vector

    return maximo

def condMC(A, p, Np=1000000):
    """
    Estima el número de condición de A (||A|| * ||A^-1||) usando Monte Carlo.
    
    Parámetros:
    A: Matriz inversible.
    p: Norma a utilizar para la estimación.
    Np: Número de muestras para Monte Carlo.
    
    Retorna estimación del número de condición o None si A no es inversible.
    """
    AInv = np.linalg.inv(A)
    if AInv is None:
        return None
    
    normaAInv = normaMatMC(AInv, p, p, Np)[0]
    normaA = normaMatMC(A, p, p, Np)[0]

    return normaA * normaAInv

def condExacta(A, p):
    """
    Calcula el número de condición exacto de A para normas 1 o infinito.
    
    Requiere calcular la inversa de A explícitamente.
    
    Retorna ||A||_p * ||A^-1||_p.
    """
    AInv = inversa(A)
    normaA = normaExacta(A, p)
    normaAInv = normaExacta(AInv, p)
    
    if normaA is None:
        return 0
    
    return normaA * normaAInv

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

    return productoMatricial(Uinv, Linv)

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

    Vt, D, nops2 = calculaLU(traspuesta(U))


    if Vt is None:
        return None, None, None, 0
    
    return L, D, traspuesta(Vt), nops1 + nops2

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
            qMoño_j = restaVectorial(qMoño_j, productoEscalar(q_k, R[k][j]))
            nops += 2*m
        
        R[j][j] = norma(qMoño_j, 2)
        nops += 2*m - 1

        if R[j][j] > tol:

            Q[:, j] = productoEscalar(qMoño_j, 1/R[j][j])
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

        valor_intermedio = productoMatricial(v_fila, R[k:, k:]).flatten()
        R[k:, k:] -= 2 * np.outer(v, valor_intermedio)
        v_columna = v.reshape(-1, 1)
        
        # Qv (m, n-k) x (n-k, n-k)
        valor_intermedio_Q = productoMatricial(Q[:, k:], v_columna).flatten()
        
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
    resta = normalizarVector(restaVectorial(colCanonico(n,0), v1),2)
    producto = productoExterno(resta, resta)  
    Hv1 = restaMatricial(nIdentidad(n), productoEscalar(producto, 2))
    mid = productoMatricial(Hv1,productoMatricial(A,traspuesta(Hv1)))

    if n == 2:
        return Hv1, mid
    
    Amoño = submatriz(mid, 2, n)
    Smoño, Dmoño = diagRH(Amoño, tol, K)

    D = extenderConIdentidad(Dmoño, n)
    D[0][0] = l1

    S = productoMatricial(Hv1, extenderConIdentidad(Smoño, n))

    return S, D

def transiciones_al_azar_continuas(n):
    """
    Genera una matriz nxn con valores aleatorios continuos y columnas normalizadas.
    """
    t = []
    for i in range(n):
        randvec = np.random.uniform(0, 1, n)
        t.append(randvec)
    tnormalizado = normaliza(t, 1)
    return traspuesta(tnormalizado)


def transiciones_al_azar_uniformes(n,thres):
    """
    Genera una matriz nxn con valores aleatorios continuos y columnas normalizadas.
    Donde el elemento (i,j) es distinto de cero si el numero generado al azar para (i,j)
    es menor o igual a 'thres'.
    """
    if n == 1:
        return np.array([[1]])

    t = []
    for i in range(n):
        randvec = np.random.uniform(0, 1, n)
        t.append(randvec)

    for i in range(len(t)):
        for j in range(len(t[0])):
            if t[i][j] < thres:
                t[i][j] = 1
            else:
                t[i][j] = 0
            if i == j:
                t[i][i] = 1
    tnormalizado = normaliza(t, 1)
    return np.array(traspuesta(tnormalizado))

#funciona pq λi es autovalor sii σi es valor singular
def nucleo(A,tol=1e-15):
    """
    Calcula una base del núcleo de A utilizando SVD/Diagonalización.
    """
    normalA = productoMatricial(traspuesta(A), A)
    SA, DA = diagRH(normalA)
    nucleo = []
    
    #consigo la columna respectiva del autovalor 0 
    for i in range(len(DA)):
            if DA[i][i] <= tol:
                nucleo.append(SA[:, i])
                
    return traspuesta(np.array(nucleo))


def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    """
    Crea una representación rala (diccionario) de una matriz a partir de una lista [filas, cols, valores].
    """
    if len(listado) == 0:
        #cualquiera pero los tests esperan esto
        return [], (m_filas, n_columnas)
    
    aristas = {}

    for i in range(len(listado[0])):
        ij_valor = listado[2][i]
        if ij_valor > tol:
            ij = ((listado[0][i]),listado[1][i])
            aristas[ij] =  ij_valor
    return aristas, (m_filas, n_columnas)

def multiplica_rala_vector(A,v):
    """
    Realiza el producto matriz-vector donde A es una matriz rala (diccionario) y v es un vector.
    """
    w = np.zeros(v.shape)
    ijs = A.keys()
    
    for parIj in ijs:
        w[parIj[0]] += A[parIj] * v[parIj[1]]

    return w

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
        A = traspuesta(A)
        usar_traspuesta = True

    m, n = A.shape

    AtA = productoMatricial(traspuesta(A), A)
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

    B = productoMatricial(A, VHat_k)
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