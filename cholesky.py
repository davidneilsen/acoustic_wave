import numpy as np

'''

THIS NEEDS TO BE DEBUGGED

'''

def penta(xu, xl, is_, ie, ns, ne, nt, albed, albef, alpha, alphf, beta, betf):
    """
    Perform LU decomposition of a pentadiagonal matrix A (stored in banded form).
    
    Parameters:
    xu : ndarray of shape (lim+1, 3), upper matrix
    xl : ndarray of shape (lim+1, 2), lower matrix
    is_, ie : int, loop bounds
    ns, ne, nt : int, indexing and case flags
    albed, albef : ndarray of shape (5, 3, 2), banded form of matrix A
    alpha, alphf, beta, betf : float, parameters for matrix entries
    """
    one = 1.0
    lim = xu.shape[0] - 1

    if nt == 0:
        albe = albed
        alpho = alpha
        beto = beta
    else:
        albe = albef
        alpho = alphf
        beto = betf

    xl[is_:ie+1, :] = one
    xu[is_:ie+1, :] = one

    i = is_
    xu[i, 0] = one
    xu[i, 1] = albe[1, 0, ns]
    xu[i, 2] = albe[2, 0, ns]

    i = is_ + 1
    xl[i, 1] = albe[-1 + 2, 1, ns] * xu[i - 1, 0]
    xu[i, 0] = one / (one - xu[i - 1, 1] * xl[i, 1])
    xu[i, 1] = albe[1, 1, ns] - xu[i - 1, 2] * xl[i, 1]
    xu[i, 2] = albe[2, 1, ns]

    i = is_ + 2
    xl[i, 0] = albe[-2 + 2, 2, ns] * xu[i - 2, 0]
    xl[i, 1] = (albe[-1 + 2, 2, ns] - xu[i - 2, 1] * xl[i, 0]) * xu[i - 1, 0]
    xu[i, 0] = one / (one - xu[i - 2, 2] * xl[i, 0] - xu[i - 1, 1] * xl[i, 1])
    xu[i, 1] = albe[1, 2, ns] - xu[i - 1, 2] * xl[i, 1]
    xu[i, 2] = albe[2, 2, ns]

    for i in range(is_ + 3, ie + 1):
        xl[i, 0] = albe[0, i - is_, ns] * xu[i - 2, 0]
        xl[i, 1] = (albe[1, i - is_, ns] - xu[i - 2, 1] * xl[i, 0]) * xu[i - 1, 0]
        xu[i, 0] = one / (one - xu[i - 2, 2] * xl[i, 0] - xu[i - 1, 1] * xl[i, 1])
        xu[i, 1] = albe[3, i - is_, ns] - xu[i - 1, 2] * xl[i, 1]
        xu[i, 2] = albe[4, i - is_, ns]



def solve_penta(xu, xl, b):
    """
    Solve A·x = b using precomputed LU decomposition of pentadiagonal matrix.
    Parameters:
        xu : ndarray of shape (N, 3), U matrix
        xl : ndarray of shape (N, 2), L matrix
        b  : ndarray of shape (N,), right-hand side
    Returns:
        x  : ndarray of shape (N,), solution to A·x = b
    """
    N = len(b)
    y = np.zeros_like(b)
    x = np.zeros_like(b)

    # Forward substitution: solve L·y = b
    y[0] = b[0]
    y[1] = b[1] - xl[1, 1] * y[0]
    for i in range(2, N):
        y[i] = b[i] - xl[i, 0] * y[i - 2] - xl[i, 1] * y[i - 1]

    # Backward substitution: solve U·x = y
    x[-1] = y[-1] / xu[-1, 0]
    x[-2] = (y[-2] - xu[-2, 1] * x[-1]) / xu[-2, 0]
    for i in reversed(range(N - 2)):
        x[i] = (y[i] - xu[i, 1] * x[i + 1] - xu[i, 2] * x[i + 2]) / xu[i, 0]

    return x
