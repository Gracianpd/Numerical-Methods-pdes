import argparse
import sys
from metodo_QR import *
from metodo_potencia import *
from soporte_plots import *

'''
Fichero principal que maneja el resto del codigo
'''
def main():
    p = argparse.ArgumentParser(description="Ejecutable para métodos numéricos (potencia / qr / analisis).")
    p.add_argument("--mode", "-m", choices=["potencia", "qr", "analisis"], required=True)
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--tol", type=float, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--csv", type=str, help="Ruta al CSV de convergencia (opcional).")
    p.add_argument("--ref", type=float, help="Valor de referencia para el plot (opcional).")
    a = p.parse_args()

    A, a_init, b_init = construir_matriz(a.dim, a.lam)

    if a.mode == "potencia":
        print("[INFO] Ejecutando método de potencia...")
        r1 = metodo_potencia_aitken(A, a.dim, a.tol, a.N)
        r2 = metodo_potencia_simetrica_aitken(A, a.dim, a.tol, a.N)
        write_potencia(r1, r2)
        if a.csv and a.ref:
            plot_convergencia(a.csv, a.ref)

    elif a.mode == "qr":
        print("[INFO] Ejecutando QR con shift...")
        vals = QR_tridiagonal_algorithm(a_init, b_init, a.tol, a.N, True)
        write_csv(vals, autovalores_exactos(a.dim, a.lam))

    else:  # analisis
        print("[INFO] Ejecutando análisis de estabilidad...")
        analizar_estabilidad_potencia_vs_teorico(a.dim)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

