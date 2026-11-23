import numpy as np

def crear_malla(L=1.0, N_div_x=10, T_final=0.1, N_div_t=200):
    """
    Construye la malla espacial y temporal.
    - N_div_x: número de divisiones espaciales (intervalos).
    - N_div_t: número de divisiones temporales (intervalos).
    - inputs: tamaño L, numero div, tiempo final, N div temporales
    -salidas: mallados x,t
    """
    x = np.linspace(-L/2, L/2, N_div_x + 1)     # N_div_x+1 nodos
    t = np.linspace(0.0, T_final, N_div_t + 1)  # N_div_t+1 instantes
    return x, t

def diffusion_numeric(x, t, D=1.0, C=0.0):
    """
    Solución numérica de ec difusion
    con CC Dirichlet n = 0 en los extremos de x.

    Condición inicial: delta en x=0 aprox discreta
    -inputs: mallados y parametros
    -output: soluciones numericas
    """
    Nx = len(x)
    Nt = len(t) - 1

    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Factor de estabilidad
    r = D * dt / dx**2
    if r > 0.5:
        raise ValueError(
            f"Esquema explícito inestable: D*dt/dx^2 = {r:.3f} > 1/2. "
            "Reduce dt o aumenta el número de divisiones temporales."
        )



    n = np.zeros(Nx)
    # nodo central
    i_center = Nx // 2
    n[i_center] = 1.0 / dx

    n_num = np.zeros((Nt + 1, Nx))
    n_num[0, :] = n

    for k in range(1, Nt + 1):
        n_new = n.copy()

        # discretizado;
        # n_i^{j+1} = n_i^j + r (n_{i+1}^j - 2 n_i^j + n_{i-1}^j) + C dt n_i^j
        n_new[1:-1] = (
            n[1:-1]
            + r * (n[2:] - 2.0 * n[1:-1] + n[:-2])
            + C * dt * n[1:-1]
        )

        # Condiciones de contorno
        n_new[0] = 0.0
        n_new[-1] = 0.0

        n = n_new
        n_num[k, :] = n

    return n_num


def diffusion_cn(x, t, D=1.0, C=0.0):
    """
    Solución numérica de digusion
    usando el metodo CN

    BCs: Dirichlet homogéneas n = 0 en los extremos de x.

    Condición inicial: delta en x=0 aprox discreta:

    Inputs: mallado espacial x y temporal t

    Output: solucionesn  n_num
    """
    Nx = len(x)
    Nt = len(t) - 1

    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Número de nodos interiores (excluimos las fronteras i=0 y i=Nx-1)
    M = Nx - 2
    if M < 1:
        raise ValueError("Se necesitan al menos 3 nodos espaciales (Nx >= 3).")

    # Parámetros del esquema
    lam = D * dt / dx ** 2  # λ = D k / h^2
    gamma = C * dt  # γ = C k

    # Matriz A (lado izquierdo): A n^{j+1} = B n^{j}
    # Coeficientes tridiagonales constantes:
    # diag(A): 1 + λ - γ/2
    # off(A):  - λ/2
    a = np.full(M - 1, -lam / 2.0)  # subdiagonal
    b = np.full(M, 1.0 + lam - gamma / 2.0)  # diagonal
    c = np.full(M - 1, -lam / 2.0)  # superdiagonal

    # Matriz B (lado derecho) implícita: B n^{j}
    # diag(B): 1 - λ + γ/2
    # off(B):   λ/2
    diag_B = 1.0 - lam + gamma / 2.0
    off_B = lam / 2.0

    # Pre-factorización LU tridiagonal de A (algoritmo de Thomas)
    # Guardamos la diagonal modificada bp y la superdiagonal modificada cp
    bp = np.empty(M)
    cp = np.empty(M - 1)

    bp[0] = b[0]
    cp[0] = c[0] / bp[0]
    for i in range(1, M - 1):
        bp[i] = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / bp[i]
    bp[M - 1] = b[M - 1] - a[M - 2] * cp[M - 2]

    # Condición inicial: delta discreta en el nodo central
    n = np.zeros(Nx)
    i_center = Nx // 2
    n[i_center] = 1.0 / dx  # para que la integral ≈ 1

    # Array para guardar toda la evolución
    n_num = np.zeros((Nt + 1, Nx))
    n_num[0, :] = n

    # Bucle temporal
    for j in range(1, Nt + 1):
        # Vector interior en el instante actual (excluimos las fronteras)
        n_int = n[1:-1]  # longitud M

        # Construimos el lado derecho d = B n^{j} (solo interior)
        d = np.empty(M)
        for i in range(M):
            val = diag_B * n_int[i]
            if i > 0:
                val += off_B * n_int[i - 1]
            if i < M - 1:
                val += off_B * n_int[i + 1]
            d[i] = val

        # Resolver A n_int_new = d con la factor. LU precomputada
        # 1) Sustitución hacia adelante: L y = d
        y = np.empty(M)
        y[0] = d[0] / bp[0]
        for i in range(1, M):
            y[i] = (d[i] - a[i - 1] * y[i - 1]) / bp[i]

        # 2) Sustitución hacia atrás: U n_int_new = y
        n_int_new = np.empty(M)
        n_int_new[-1] = y[-1]
        for i in range(M - 2, -1, -1):
            n_int_new[i] = y[i] - cp[i] * n_int_new[i + 1]

        # Reconstruimos el vector completo incluyendo fronteras
        n_new = np.zeros(Nx)
        n_new[1:-1] = n_int_new
        # n_new[0] y n_new[-1] ya son 0 (Dirichlet)

        n = n_new
        n_num[j, :] = n

    return n_num

def diffusion_exact(x, t, D=1.0, C=0.0, N_modes=200):
    """
    Solución exacta para ec difusion, dirichlet, CI delta
    (solo contribuyen los términos impares).
    inputs: mallado x,t parametros C, D numero de puntos de la serie que calcualr
    """
    Nx = len(x)
    Nt = len(t) - 1
    L = x[-1] - x[0]

    n_exact = np.zeros((Nt + 1, Nx))

    # Precalculamos cosas que no dependen de t
    n_idx = np.arange(1, N_modes + 1)
    lam_n = (n_idx * np.pi / L) ** 2
    a_n = (2.0 / L) * np.sin(n_idx * np.pi / 2.0)  # coeficientes iniciales

    # Matriz de autofunciones sin(nπ(x+L/2)/L) para todos los x y n
    # shape: (N_modes, Nx)
    phi = np.sin(np.outer(n_idx, (x + L/2) * np.pi / L))

    for k, tk in enumerate(t):
        # coeficientes temporales e^{(C - D λ_n) t}
        time_factor = np.exp((C - D * lam_n) * tk)
        # combinación lineal: Σ_n a_n * time_factor_n * phi_n(x)
        n_exact[k, :] = (a_n * time_factor) @ phi

    return n_exact

def calcular_error_espacio(N_div_x, N_div_t, T_final, D=1.0, C=0.0, L=1.0, N_modes=800):
    """
    Fija el paso temporal (N_div_t) y el tiempo final T_final.
    Varía N_div_x (h) y calcula el error a t = T_final.
    input: divisiones temporales y espaciales, tiempo final, parametros
    Output: h, error_L2, error_inf
    """
    # Malla para este N_div_x con el mismo N_div_t
    x, t = crear_malla(L=L, N_div_x=N_div_x, T_final=T_final, N_div_t=N_div_t)

    # Soluciones
    n_num = diffusion_numeric(x, t, D=D, C=C)
    n_exa = diffusion_exact(x, t, D=D, C=C, N_modes=N_modes)

    # Error a tiempo final
    dx = x[1] - x[0]
    error = np.abs(n_num[-1, :] - n_exa[-1, :])

    error_inf = np.max(error)
    error_L2  = np.sqrt(np.sum(error**2) * dx)
    h = dx

    return h, error_L2, error_inf

def densidad_promedio(x, n_xt, solo_mitad=True):
    """
    Calcula la densidad 'promedio' de neutrones n(t) = ∫ n(x,t) dx

    Inputs:
    x : array (Nx,)
        Nodos espaciales.
    n_xt : array (Nt+1, Nx)
        Solución en función de t y x. Se asume n_xt[j, i] = n(x_i, t_j).
    solo_mitad : bool
        Si True, integra solo en x >= 0 y multiplica por 2
        (usando la simetría). Si False, integra en todo el dominio.

    Outputs:
    n_t :
        Valores de n(t) para cada instante temporal.
    """
    dx = x[1] - x[0]

    if solo_mitad:
        mask_pos = x >= 0.0
        # integral sobre [0, L/2]
        n_med = np.sum(n_xt[:, mask_pos], axis=1) * dx
        # aprovechamos la simetría: ∫_{-L/2}^{L/2} = 2 ∫_0^{L/2}
        n_t = n_med
    else:
        # integral en todo el dominio
        n_t = np.sum(n_xt, axis=1) * dx

    return n_t

def calcular_error_espacio_cn(N_div_x, N_div_t, T_final, D=1.0, C=0.0, L=1.0, N_modes=800):
    """
    Igual que calcular_error_espacio, pero usando el CN
    """
    x, t = crear_malla(L=L, N_div_x=N_div_x, T_final=T_final, N_div_t=N_div_t)

    n_num = diffusion_cn(x, t, D=D, C=C)
    n_exa = diffusion_exact(x, t, D=D, C=C, N_modes=N_modes)

    dx = x[1] - x[0]
    error = np.abs(n_num[-1, :] - n_exa[-1, :])

    error_inf = np.max(error)
    error_L2  = np.sqrt(np.sum(error**2) * dx)
    h = dx

    return h, error_L2, error_inf

def calcular_error_tiempo_prog(N_div_x, N_div_t, T_final,
                               D=1.0, C=0.0, L=1.0, N_modes=800):
    """
    Fija la malla espacial (N_div_x) y el tiempo final T_final.
    Varía N_div_t (k = T_final / N_div_t) y calcula el error a t = T_final
    Ppara prog
    Devuelve:
        dt, error_L2
    """
    x, t = crear_malla(L=L, N_div_x=N_div_x, T_final=T_final, N_div_t=N_div_t)

    n_num = diffusion_numeric(x, t, D=D, C=C)
    n_exa = diffusion_exact(x, t, D=D, C=C, N_modes=N_modes)

    dx = x[1] - x[0]
    error = np.abs(n_num[-1, :] - n_exa[-1, :])

    error_L2 = np.sqrt(np.sum(error**2) * dx)
    dt = t[1] - t[0]
    return dt, error_L2


def calcular_error_tiempo_cn(N_div_x, N_div_t, T_final,
                             D=1.0, C=0.0, L=1.0, N_modes=800):
    """
    Igual que calcular_error_tiempo_prog pero usando CN
    Devuelve:
        dt, error_L2
    """
    x, t = crear_malla(L=L, N_div_x=N_div_x, T_final=T_final, N_div_t=N_div_t)

    n_num = diffusion_cn(x, t, D=D, C=C)
    n_exa = diffusion_exact(x, t, D=D, C=C, N_modes=N_modes)

    dx = x[1] - x[0]
    error = np.abs(n_num[-1, :] - n_exa[-1, :])

    error_L2 = np.sqrt(np.sum(error**2) * dx)
    dt = t[1] - t[0]
    return dt, error_L2
