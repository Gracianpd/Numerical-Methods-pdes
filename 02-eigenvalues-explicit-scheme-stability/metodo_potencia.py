import numpy as np

'''
Metodo potencia y potencia simetrica, con y sin Aitken
'''
def metodo_potencia_aitken(A, n, TOL, N):
    #funcion que calcula autoval dominante y su autovect. Metodo normal y acelerado aitken
    #entradas: matriz A, dimension n, Tolerancia TOL y numero maximo de iteraciones N
    #salidas: vector con, para cada iteración, numero iteracion, autovalor, autovalor aitken, autovector
    x = np.ones(n)
    p = np.argmax(np.abs(x))
    x = x / x[p]


    mu_hist = []
    results = []

    for k in range(1, N + 1):

        y = A @ x
        p = np.argmax(np.abs(y))
        yp = y[p]

        if yp == 0:
            results.append((k, None, None, x.copy()))
            return results

        μ = yp
        mu_hist.append(μ)

        μ_aitken = None
        if len(mu_hist) >= 3:
            μ0, μ1, μ = mu_hist[-3:]
            denom = μ - 2*μ1 + μ0
            if abs(denom) > 1e-14:
                μ_aitken = μ0 - (μ1 - μ0)**2 / denom

        x_new = y / yp
        err = np.max(np.abs(x_new - x))
        x = x_new

        results.append((k, μ, μ_aitken, x.copy()))

        if err < TOL:
            break

    return results


def metodo_potencia_simetrica_aitken(A, n, TOL, N):
    #metodo potencia simetrica. Metodo de la potencia para matriz simetrica.
    #mismas entradas y salidas que funcion anterior
    x = np.ones(n)
    x = x / np.linalg.norm(x)

    mu_hist = []
    results = []

    for k in range(1, N + 1):

        y = A @ x
        μ = float(x @ y)
        mu_hist.append(μ)

        y_norm = np.linalg.norm(y)
        if y_norm == 0:
            results.append((k, None, None, x.copy()))
            return results

        μ_aitken = None
        if len(mu_hist) >= 3:
            μ0, μ1, μ = mu_hist[-3:]
            denom = μ - 2*μ1 + μ0
            if abs(denom) > 1e-14:
                μ_aitken = μ0 - (μ1 - μ0)**2 / denom

        x_new = y / y_norm
        err = np.linalg.norm(x_new - x)
        x = x_new

        results.append((k, μ, μ_aitken, x.copy()))

        if err < TOL:
            break

    return results