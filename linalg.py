import numpy as np

# ==========================================
# 1. AUX
# ==========================================

def get_canonical_row(dimension, i):
    """
    Returns the canonical vector 'i' of the respective vector space of 'dimension' as a row vector.
    """
    row = np.zeros(dimension)
    row[i] = 1
    return row

def get_canonical_col(dimension, i):
    """
    Returns the canonical vector 'i' of the respective vector space of 'dimension' as a column vector.
    """
    col = np.zeros((dimension, 1))
    col[i][0] = 1
    return col

def computer_norm(vector_input, p):
    """
    Calculates the vector p-norm of a vector input.
    
    Parameters:
    vector_input: Numpy vector or list of numbers.
    p: Order of the norm (int) or the string 'inf' for infinity norm.
    
    Returns the scalar value of the norm.
    """
    if p == 'inf':
        return max(map(abs, vector_input))
    
    result = np.sum(np.abs(vector_input) ** p)
    return result**(1/p)

def normalize_vector(vector, p):
    """
    Returns the vector passed as a parameter normalized in p-norm.
    """
    vector_norm = computer_norm(vector, p)
    
    if vector_norm == 0:
        return vector
    
    return np.array(vector) / vector_norm

def normalize_vector_list(vectors_list, p):
    """
    Normalizes a list of vectors according to the indicated p-norm.
    
    Parameters:
    vectors_list: List of vectors.
    p: Order of the norm to be used.
    
    Returns a list of unit vectors.
    """
    normalized_list = []

    for vector in vectors_list:
        res = normalize_vector(vector, p)
        normalized_list.append(res)

    return normalized_list

def get_sign(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0

def compute_Ax(A, x):
    """
    Calculates the matrix-vector product with parameters A and x.
    """
    x_flat = np.asarray(x).flatten()
    res = (A @ x_flat).astype(float)
    
    return res.reshape(-1, 1)

def get_upper_triangular(A):
    """
    Returns matrix A but with 0s below the diagonal.
    """
    upper_tri = A.copy()

    for i in range(len(A)):
        for j in range(len(A[i])):
            if j < i:
                upper_tri[i, j] = 0
    
    return upper_tri

def get_lower_triangular(A):
    """
    Returns matrix A but with 0s above the diagonal and rewriting the diagonal with 1s.
    """
    lower_tri = A.copy()

    for i in range(len(A)):
        for j in range(len(A[i])):
            if j > i:
                lower_tri[i][j] = 0
            if j == i:
                lower_tri[i][i] = 1
    
    return lower_tri

def extend_with_identity(A, p): 
    """
    Returns matrix A extended to the top-left with 1s on the diagonal.
    """
    n = A.shape[0]
    res = np.eye(p)
    res[p-n:, p-n:] = A
    return res

def get_submatrix(A, l, k):
    """
    Crops matrix A from A[l][l] to A[k][k].
    """
    return A[l-1:k, l-1:k]

def get_column_suffix(A, j, k):
    """
    Extracts column j of matrix A, considering only indices from k onwards.
    """
    return A[k:A.shape[1], j]

def is_symmetric(A, tol=1e-10): 
    """
    Returns True if matrix A is symmetric under the tolerance 'tol'.
    """
    return np.allclose(A, A.T, tol)

# ==========================================
# 2. TRIANGULAR SOLVERS
# ==========================================

def backward_substitution(A, b):
    """
    Solves the linear system Ax = b where A is upper triangular.
    Returns the solution vector x.
    """
    m, n = A.shape
    solution_x = np.zeros(n)

    for i in range(min(m, n) - 1, -1, -1):
        current_divisor = A[i][i]

        if current_divisor == 0:
            solution_x[i] = np.nan
        else:
            summation = np.dot(A[i, i+1:], solution_x[i+1:])
            solution_x[i] = (b[i] - summation) / current_divisor

    return solution_x

def forward_substitution(A, b):
    """
    Solves the linear system Ax = b where A is lower triangular.
    Returns the solution vector x.
    """
    m, n = A.shape
    solution_x = np.zeros(n)

    for i in range(min(m, n)):
        current_divisor = A[i][i]
        
        if current_divisor == 0:
            solution_x[i] = np.nan
        else:
            summation = np.dot(A[i, :i], solution_x[:i])
            solution_x[i] = (b[i] - summation) / current_divisor
    return solution_x

def solve_triangular(L, b, lower=True):
    """
    Wrapper for solving triangular systems.
    """
    if lower:
        return forward_substitution(L, b)
    return backward_substitution(L, b)

# ==========================================
# 3. CHOLESKY & LU
# ==========================================

def compute_LU(A):
    """
    Calculates the LU decomposition of matrix A without pivoting.
    """
    m, n = A.shape
    Ac = A.copy()
    
    if m != n:
        return None, None

    for k in range(0, n-1):
        if Ac[k][k] == 0:
            return None, None
        
        for i in range(k + 1, m):
            quotient = Ac[i][k] / Ac[k][k]
            Ac[i][k] = quotient
            Ac[i, k+1 : n] = Ac[i, k+1 : n] - quotient * Ac[k, k+1 : n]

    return get_lower_triangular(Ac), get_upper_triangular(Ac)

def invert_LU(A):
    """
    Calculates the inverse of A using LU decomposition.
    """
    n = A.shape[0]

    L, U = compute_LU(A)

    if L is None or U is None:
        return None

    Linv = np.zeros((n, n))
    Uinv = np.zeros((n, n))

    for i in range(n):
        col_inv = solve_triangular(L, get_canonical_row(n, i), lower=True)
        for j in range(n):
            Linv[j][i] = col_inv[j]

    for i in range(n):
        if U[i, i] == 0:
            return None

        col_inv = solve_triangular(U, get_canonical_row(n, i), lower=False)
        for j in range(n):
            Uinv[j][i] = col_inv[j]

    return Uinv @ Linv

def compute_LDV(A):
    """
    Calculates the L D V^T decomposition.
    """
    L, U = compute_LU(A)

    if U is None:
        return None, None, None

    Vt, D = compute_LU(U.T)

    if Vt is None:
        return None, None, None
    
    return L, D, Vt.T

def is_SPD(A, atol=1e-10):
    """
    Determines if matrix A is Symmetric Positive Definite (SPD).
    """
    if not is_symmetric(A, atol):
        return False
    
    L, D, Lt = compute_LDV(A)

    if D is None:
        return False
    
    for i in range(len(D)):
        if D[i, i] <= 0:
            return False
    return True

def compute_cholesky(A):
    """
    Calculates matrix L from the Cholesky decomposition.
    """
    if not is_SPD(A):
        return None

    L, D, _ = compute_LDV(A)

    for i in range(len(D)): 
        D[i][i] = np.sqrt(D[i][i])

    return L @ D

# ==========================================
# 4. ORTHOGONALIZATION & QR
# ==========================================

def qr_gram_schmidt(A, tol=1e-10):
    """
    Calculates QR factorization using the Gram-Schmidt process.
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    a_1 = A[:, 0]
    norm1 = computer_norm(a_1, 2)
    R[0][0] = norm1

    if norm1 > tol:
        Q[:, 0] = normalize_vector(a_1, 2)
    else:
        Q[:, 0] = a_1

    for j in range(1, n):
        q_tilde_j = A[:, j]

        for k in range(0, j):
            q_k = Q[:, k]
            R[k][j] = np.vdot(q_k, q_tilde_j)
            q_tilde_j = q_tilde_j - (q_k * R[k][j])
        
        R[j][j] = computer_norm(q_tilde_j, 2)

        if R[j][j] > tol:
            Q[:, j] = q_tilde_j * 1/R[j][j]
        else:
            Q[:, j] = q_tilde_j

    return Q, R

def qr_householder(A, tol=1e-10):
    """
    Calculates QR factorization using Householder reflections.
    """
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)

    if m < n:
        return None, None

    for k in range(min(m, n)):
        x = R[k:, k]
        
        norm_x = computer_norm(x, 2)
        if norm_x < tol:
            continue
            
        sign_x = get_sign(x[0])
        u = x.copy()
        u[0] += sign_x * norm_x
        
        v = u / computer_norm(u, 2)
        v_row = v.reshape(1, -1)

        intermediate_val = (v_row @ R[k:, k:]).flatten()
        R[k:, k:] -= 2 * np.outer(v, intermediate_val)
        v_col = v.reshape(-1, 1)
        
        intermediate_val_Q = (Q[:, k:] @ v_col).flatten()
        Q[:, k:] -= 2 * np.outer(intermediate_val_Q, v)

    return Q, R

def compute_qr(A, method='HH', tol=1e-10):
    """
    Calculates the QR decomposition of A.
    """
    if method == 'HH':
        return qr_householder(A, tol)
    elif method == 'GS':
        return qr_gram_schmidt(A, tol)
    else: 
        return None, None

# ==========================================
# 5. DIAGONALIZATION & SVD
# ==========================================

def diagonalize_HH(A, tol=1e-10, K=100):
    """
    Performs recursive Householder diagonalization.
    """
    n = len(A)

    v1, l1, _ = power_method(A, tol, K)
    u = normalize_vector((get_canonical_col(n, 0) - v1), 2).flatten()
    Au = A @ u
    uAu = np.dot(u, Au)
    q = Au - uAu * u
    W_outer = np.outer(q, u)
    
    mid = A - 2 * (W_outer + W_outer.T)

    if n == 2:
        uut = np.outer(u, u)
        Anew = np.eye(n) - 2 * uut
        return Anew, mid
    
    A_tilde = get_submatrix(mid, 2, n)
    S_tilde, D_tilde = diagonalize_HH(A_tilde, tol, K)

    D_matrix = extend_with_identity(D_tilde, n)
    D_matrix[0][0] = l1

    S_tilde_ext = extend_with_identity(S_tilde, n)
    S_matrix = S_tilde_ext - 2 * np.outer(u, u @ S_tilde_ext)

    return S_matrix, D_matrix

def get_singular_values_vector(sigma_hat, k):
    """
    Extracts and calculates singular values from the diagonal eigenvalue matrix.
    """
    sigma_hat_vector = []
    for i in range(k):
            sigma_hat_vector.append(np.sqrt(np.abs(sigma_hat[i][i])))
    return sigma_hat_vector

def reduced_SVD(A, k="max", tol=1e-10):
    """
    Calculates the reduced Singular Value Decomposition (SVD) of A.
    """
    m, n = A.shape

    use_transpose = False
    if m < n:
        A = A.T
        use_transpose = True

    m, n = A.shape

    AtA = (A.T) @ A
    v_hat_full, sigma_hat = diagonalize_HH(AtA, tol=tol, K=100)

    rank = min(m, n)
    for i in range(len(sigma_hat)):
        if sigma_hat[i, i] < tol:
            rank = i
            break
    rank = min(m, n, rank)
    k = rank if k == "max" else k

    v_hat_k = v_hat_full[:, :k]
    sigma_hat_vector = get_singular_values_vector(sigma_hat, k)

    B_matrix = A @ v_hat_k
    u_hat_k = B_matrix
    for j in range(k): 
        sigma_val = sigma_hat_vector[j]
        u_hat_k[:, j] /= sigma_val

    if use_transpose:
        return v_hat_k, sigma_hat_vector, u_hat_k
    else:
        return u_hat_k, sigma_hat_vector, v_hat_k

def truncate_SVD(U, S, V):
    """
    Calculates the reduced version of the SVD from the full version.
    """
    U = np.array(U) if isinstance(U, list) else U
    V = np.array(V) if isinstance(V, list) else V
    S = np.array(S) if isinstance(S, list) else S
    m, n = U.shape[0], V.shape[1]
    rank = min(m, n)

    s_list = []
    if len(S.shape) == 1: 
        s_list = S[:rank]
    elif len(S.shape) == 2: 
        s_list = [S[i, i] for i in range(rank)]
    else:
        raise TypeError(f'Invalid S argument: must be a list or a diagonal matrix of singular values.')

    U_red = U[:, :rank]
    V_red = V[:, :rank]

    return U_red, s_list, V_red

def power_method_func(A, v):
    """
    Auxiliary function for the power method.
    """
    w_prime = compute_Ax(A, v)

    if computer_norm(w_prime, 2) > 0:
        return normalize_vector(w_prime, 2)

    return np.zeros_like(w_prime)

def power_method(A, tol=1e-10, K=100.0):
    """
    Calculates the dominant eigenvalue and its associated eigenvector using the power method.
    """
    n = len(A[0])
    v_vec = np.random.rand(n, 1)
    v_tilde_temp = power_method_func(A, v_vec)
    v_tilde = power_method_func(A, v_tilde_temp)
    e_val = float(np.vdot(v_tilde, v_vec))
    k_iter = 0
    while(1 - abs(e_val) > tol and k_iter < K):
        v_vec = v_tilde
        v_tilde = power_method_func(A, v_tilde)
        e_val = float(np.vdot(v_tilde, v_vec))
        k_iter = k_iter + 1
    
    ax_vec = compute_Ax(A, v_tilde)
    eigenvalue = np.vdot(v_tilde, ax_vec)

    return v_tilde, eigenvalue, k_iter