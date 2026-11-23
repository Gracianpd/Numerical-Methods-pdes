import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from metodo_potencia import *

'''
Funciones de soporte para calculos y graficas
'''
def construir_matriz(N, lam):
    #construye matriz del problema
    #entrada: dimension, parametro
    # matriz completa, diagonal principal y inferior/superior
    a = np.full(N, 1 - 2 * lam, dtype=float)

    b = np.full(N - 1, lam, dtype=float)
    A = np.zeros((N, N), dtype=float)

    for i in range(N):
        A[i, i] = a[i]
        if i < N - 1:
            A[i, i + 1] = lam
            A[i + 1, i] = lam

    return A, a, b

def autovalores_exactos(N, lam):
    #calcula los autovalores exactos de la matriz
    #entrada: dimensión, parametro lambda
    #salida: autovalores
    autoval = []
    for i in range(N):
        k = i +1
        val = (1-2 * lam) + 2 * lam * np.cos((k * np.pi) / (N + 1))
        autoval.append(val)
    return autoval

def write_csv(qr, exact):
    #escribe para el metodo qr en un csv los resultados
    norm = []
    for x in qr:
        if isinstance(x, list):
            norm.append(x)
        else:
            norm.append([x, None])

    with open("output.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Value1", "Value2", "Integer"])
        for (v1, num), v2 in zip(norm, exact):
            writer.writerow([v1, v2, num])

def write_potencia(res_norm, res_sim):
    #escribe para el metodo de la potencia en un csv los resultados
    #entrada: vectores de potencia normal y simetrica, cony sin aitken
    max_len = max(len(res_norm), len(res_sim))
    res_norm += [(None, None, None, None)] * (max_len - len(res_norm))
    res_sim += [(None, None, None, None)] * (max_len - len(res_sim))

    with open("potencia.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "mu", "mu_aitken", "mu_simetrica", "mu_simetrica_aitken"])

        for i in range(max_len):
            k_norm, mu_norm, mu_aitken, _ = res_norm[i]
            k_sim, mu_sim, mu_sim_aitken, _ = res_sim[i]

            # Usa el k válido (de cualquiera de los dos)
            k = k_norm if k_norm is not None else k_sim

            writer.writerow([k, mu_norm, mu_aitken, mu_sim, mu_sim_aitken])

def plot_convergencia(csv_path: str, lambda1: float):
    #entada: csv con datos de metodo de potencia
    #salida: graficas
    df = pd.read_csv(csv_path)

    def calc_error(col):
        return abs(df[col] - lambda1) if col in df.columns else np.nan

    df["err_mu"] = calc_error("mu")
    df["err_mu_sim"] = calc_error("mu_simetrica")
    df["err_mu_aitken"] = calc_error("mu_aitken")
    df["err_mu_sim_aitken"] = calc_error("mu_simetrica_aitken")

    def calc_ratio(err_col):
        e = err_col.values.astype(float)
        r = [np.nan]
        for i in range(1, len(e)):
            if np.isfinite(e[i]) and np.isfinite(e[i - 1]) and e[i - 1] != 0:
                r.append(e[i] / e[i - 1])
            else:
                r.append(np.nan)
        return r

    df["ratio_mu"] = calc_ratio(df["err_mu"])
    df["ratio_mu_sim"] = calc_ratio(df["err_mu_sim"])
    df["ratio_mu_aitken"] = calc_ratio(df["err_mu_aitken"])
    df["ratio_mu_sim_aitken"] = calc_ratio(df["err_mu_sim_aitken"])

    plt.figure(figsize=(8,5))
    plt.plot(df["k"], np.log10(df["err_mu"]), "o-", label="Potencia normal")
    plt.plot(df["k"], np.log10(df["err_mu_sim"]), "s-", label="Potencia simétrica")
    plt.xlabel("Iteración k")
    plt.ylabel("log10(|μₖ - λ₁|)")
    plt.title(f"Convergencia sin Aitken (λ₁ = {lambda1:.8f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergencia_error_sin_aitken.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(df["k"], np.log10(df["err_mu_aitken"]), "o-", label="Normal Aitken")
    plt.plot(df["k"], np.log10(df["err_mu_sim_aitken"]), "s-", label="Simétrica Aitken")
    plt.xlabel("Iteración k")
    plt.ylabel("log10(|μₖ - λ₁|)")
    plt.title(f"Convergencia con Aitken (λ₁ = {lambda1:.8f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergencia_error_con_aitken.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(df["k"], df["ratio_mu"], "o-", label="Potencia normal")
    plt.plot(df["k"], df["ratio_mu_sim"], "s-", label="Potencia simétrica")
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    plt.xlabel("Iteración k")
    plt.ylabel("eₖ₊₁ / eₖ")
    plt.title("Ratios de convergencia (sin Aitken)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergencia_ratio_sin_aitken.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(df["k"], df["ratio_mu_aitken"], "o-", label="Normal Aitken")
    plt.plot(df["k"], df["ratio_mu_sim_aitken"], "s-", label="Simétrica Aitken")
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    plt.xlabel("Iteración k")
    plt.ylabel("eₖ₊₁ / eₖ")
    plt.title("Ratios de convergencia (con Aitken)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergencia_ratio_con_aitken.png", dpi=300)
    plt.show()


def analizar_estabilidad_potencia_vs_teorico(N=10, lambdas=None, TOL=1e-10, max_iter=200):
    #funcion para analizar la estabilidad del metodo xi+1=Axi
    #Entrada: N, tolerancia, maximas iteraciones
    #salida: plot
    if lambdas is None:
        lambdas = np.linspace(0, 1, 51)

    mu_dominantes = []
    mu_teoricos = []

    for lam in lambdas:
        A, a, b = construir_matriz(N, lam)
        autoval = autovalores_exactos(N, lam)
        mu_teorico = max(np.abs(autoval))   # radio espectral exacto
        mu_teoricos.append(mu_teorico)

        resultados = metodo_potencia_aitken(A, N, TOL, max_iter)
        mus = [r[1] for r in resultados if r[1] is not None]
        mu_dominante = mus[-1] if len(mus) > 0 else np.nan
        mu_dominantes.append(np.abs(mu_dominante))

    mu_dominantes = np.array(mu_dominantes, dtype=float)
    mu_teoricos = np.array(mu_teoricos, dtype=float)
    lambdas = np.array(lambdas, dtype=float)

    plt.figure(figsize=(8,5))
    plt.plot(lambdas, mu_teoricos, 's--', label='|μ|max teórico (exacto)')
    plt.plot(lambdas, mu_dominantes, 'o-', label='|μ_dominante| (método de la potencia)')
    lam_max = 1 / (1 + np.cos(np.pi / (N + 1)))
    plt.axvline(lam_max, color='r', linestyle='--', label=f'λ_max ≈ {lam_max:.4f}')
    plt.xlabel('λ')
    plt.ylabel('|μ| (autovalor dominante)')
    plt.title(f'Estudio estabilidad (N={N})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("estabilidad_metodo.png", dpi=300)
    plt.show()
