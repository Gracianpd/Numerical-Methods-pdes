import numpy as np
from scipy.linalg import solve

def gauss_seidel(A, b, x0=None, maxit=10000, tol=1e-12, check_dd=True):
    #metodo gauss seidel para calcular temperaturas en puntos
    #entradas: matriz A y vector b, vector inicial si se desea, maximo numero iteraciones, tolerancia, check a dominante diagonal
    #salidas: vector x de temperaturas
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    # Comprobaciones básicas
    if np.any(np.diag(A) == 0):
        raise ValueError("La diagonal de A contiene ceros.")
    if check_dd:
        absA = np.abs(A)
        diag = np.abs(np.diag(A))
        offdiag_sum = np.sum(absA, axis=1) - diag
        if not np.all(diag >= offdiag_sum):
            print("⚠️ Advertencia: la matriz NO es diagonalmente dominante. "
                  "El método podría no converger.")


    for _ in range(maxit):
        x_old = x.copy()

        for i in range(n):
            # sum_{j=1}^{i-1} a_ij x_j^{(k+1)}
            s1 = np.dot(A[i, :i], x[:i]) if i > 0 else 0.0
            # sum_{j=i+1}^{n} a_ij x_j^{(k)}
            s2 = np.dot(A[i, i+1:], x_old[i+1:]) if i+1 < n else 0.0

            x[i] = ( - s1 - s2 + b[i] ) / A[i, i]

        # criterio de parada por cambio máximo
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, _

    return x

def gauss_seidel_sor(A, b, omega=1.2, x0=None, maxit=10000, tol=1e-12, check_dd=True):
    #metodo gauss seidel para calcular temperaturas en puntos con metodo SOR
    #entradas: matriz A y vector b, omega para SOR, vector inicial si se desea, maximo numero iteraciones, tolerancia, check a dominante diagonal
    #salidas: vector x de temperaturas
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    # Comprobaciones básicas
    if A.shape[0] != A.shape[1]:
        raise ValueError("A debe ser cuadrada.")
    if b.shape[0] != n:
        raise ValueError("Dimensiones incompatibles entre A y b.")
    if not (1 < omega < 2):
        raise ValueError("El parámetro ω debe estar en (0,2) para SOR.")

    # Comprobación de dominancia diagonal
    if check_dd:
        absA = np.abs(A)
        diag = np.abs(np.diag(A))
        offdiag_sum = np.sum(absA, axis=1) - diag
        if not np.all(diag >= offdiag_sum):
            print("⚠️ Advertencia: la matriz NO es diagonalmente dominante. "
                  "El método podría no converger.")

    # Iteraciones SOR
    for _ in range(maxit):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i]) if i > 0 else 0.0
            s2 = np.dot(A[i, i+1:], x_old[i+1:]) if i+1 < n else 0.0
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - s1 - s2)
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, _
    return x

def laplace_semidisco(drho=1/3, dtheta=np.pi/6, Nr=4, Ntheta=7, Nseries=500):
    #solucion exacta laplace truncada a cierto N
    #entradas: punto r,theta y numero serie
    #salida resultado u

    #solo para ptos interiores
    rho = np.arange(1, Nr-1) * drho
    theta = np.arange(1, Ntheta-1) * dtheta

    # Calculamos la matriz U
    U = np.zeros((len(rho), len(theta)))
    for k in range(Nseries):
        n = 2 * k + 1
        U += (4 / np.pi) * np.outer(rho ** n, np.sin(n * theta)) / n

    # Convertimos a vect
    u_vec = U.ravel()
    return u_vec

def pass_to_dimensional(T0,T1,u):
    #pasar solucion adimensional a dimensional
    #entradas: temperaturas de CC y vector de temperaturas adimensional
    T = []
    for sol in u:
        temp = T0 + (T1 - T0) * sol
        T.append(temp)
    return T


def cartesian_to_polar(x, y):
    #funcion que pasa coordenadas de cartesianas a polares
    #entrada: veectro de puntos x e y
    #salida valores de rho y teta correspondiendtes
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)  # handles all quadrants correctly
    return rho, theta

def u_exact_from_polar(rho, theta, Nseries=1500):
    #calcula valor exacto de solucion
    #valores rho y theta donde se quiere calcular solucion y numero de elementos de la serie infita que se consideran
    #salida: valores de temperatura adimensional exactos
    rho   = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    u = np.zeros_like(rho, dtype=float)

    for m in range(Nseries):
        n = 2*m + 1
        u += (4.0 / (n*np.pi)) * (rho**n) * np.sin(n*theta)

    return u