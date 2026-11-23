from build_matrix import *
from metodos_num import *
from writer import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

def modo_simple(N_i, N_j, T0, T1, ruta_csv):
    #ejecuta el modo sencillo, calcula las T adimensionales y totales para ciertos mallados y temperaturas iniciales
    #entrada: tamaño mallados, CC y ruta salida
    #salida: csvs con T y u
    dt = np.pi / N_j
    drho = 1 / N_i

    #matrices
    A, b = build_matriz(dt, 1, N_i - 1, 1, N_j -1)
    A_cart, b_cart = build_matriz_cartesian()

    #adimensional
    u_GS, it = gauss_seidel(A, b, maxit=10000, tol=1e-12)
    u_SOR, it_SOR = gauss_seidel_sor(A, b, omega=1.2, maxit=10000, tol=1e-12)
    u_ex = laplace_semidisco()
    u_cart = gauss_seidel_sor(A_cart, b_cart)

    #temperaturas
    T_GS = pass_to_dimensional(T0, T1, u_GS)
    T_SOR = pass_to_dimensional(T0, T1, u_SOR)
    T_ex = pass_to_dimensional(T0, T1, u_ex)
    T_cart= pass_to_dimensional(T0, T1, u_cart)
    ruta_temp = ruta_csv + "temperaturas.csv"
    tura_adim = ruta_csv + "temp_adim.csv"
    exportar_temperaturas_excel(T_GS, T_SOR, T_ex, T_cart, ruta_temp)
    exportar_adimensional_excel(u_GS, it, u_SOR, it_SOR, u_ex, u_cart, tura_adim)


def varios_mallados(ruta_output):
    #estudiar multiple
    #sin entradas
    #salidas: grafica con regresion lineal
    niveles = [3, 4, 5, 6, 7, 8, 9, 10]   # número de intervalos radiales base (i)
    resultados = []

    print("\n===== ANÁLISIS DE ERROR MEDIO PARA VARIOS MALLADOS =====")

    for N_i in niveles:
        N_j = 2 * N_i
        dtheta = np.pi / N_j
        drho = 1.0 / N_i
        omega = 1.2
        tol = 1e-8

        # ---- Construcción de matrices
        A, b = build_matriz(dtheta,1, N_i - 1, 1, N_j -1)

        # ---- Solución numérica
        u_GS, it_GS = gauss_seidel(A, b, maxit=10000, tol=tol)
        u_SOR, it_SOR = gauss_seidel_sor(A, b, omega=omega, maxit=10000, tol=tol)
        u_ex = laplace_semidisco(drho=drho, dtheta=dtheta, Nr=N_i+1, Ntheta=N_j+1)

        # ---- Ajustar tamaños
        n = min(len(u_GS), len(u_ex))
        u_GS = np.array(u_GS[:n])
        u_SOR = np.array(u_SOR[:n])
        u_ex = np.array(u_ex[:n])

        # ---- Error medio absoluto
        err_GS = np.sum(np.abs(u_GS - u_ex)) / n
        err_SOR = np.sum(np.abs(u_SOR - u_ex)) / n


        resultados.append((N_i, N_j, drho, dtheta, err_GS, err_SOR))

        print(f"N_i={N_i:3d}, N_j={N_j:3d}, Δρ={drho:.4f}, Δθ={dtheta:.4f} | "
              f"Error medio GS={err_GS:.3e}, SOR={err_SOR:.3e}")


    # csv
    ruta_csv = ruta_output + "error_medio_varios_mallados.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        f.write("N_i,N_j,drho,dtheta,Error_GS,Error_SOR\n")
        for r in resultados:
            f.write(",".join(f"{x:.6e}" if isinstance(x, float) else str(x) for x in r) + "\n")

    print(f"\n✅ Resultados guardados en '{ruta_csv}'")

    drho2_dtheta2 = []
    err_GS = []
    err_SOR = []
    labels = []  # guardaremos (N_i, N_j)

    for r in resultados:
        drho2_dtheta2.append(r[2]**2 + r[3]**2)
        err_GS.append(r[4])
        err_SOR.append(r[5])
        labels.append((r[0], r[1]))

    # regresion
    x = np.array(drho2_dtheta2)
    y = np.array(err_GS)

    logx = np.log(x)
    logy = np.log(y)
    coef = np.polyfit(logx, logy, 1)
    pendiente = coef[0]
    intercepto = coef[1]
    C = np.exp(intercepto)

    # ajuste
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = C * x_fit**pendiente

    # plotting
    plt.figure(figsize=(7,5))
    plt.loglog(drho2_dtheta2, err_GS, 'o-', label=f'Gauss–Seidel (p ≈ {pendiente:.2f})', linewidth=2, markersize=6)
    plt.loglog(drho2_dtheta2, err_SOR, 's--', label='SOR (ω=1.2)', linewidth=2, markersize=6)

    # --- Etiquetas (N_i, N_j) ---
    for x_i, y_i, (Ni, Nj) in zip(drho2_dtheta2, err_GS, labels):
        plt.text(x_i, y_i*1.25, f"({Ni},{Nj})", fontsize=9, color="black",
                 ha='center', va='bottom')

    # Línea de referencia (pendiente 1)
    ref_x = np.array(drho2_dtheta2)
    ref_y = err_GS[0] * (ref_x / ref_x[0])
    plt.loglog(ref_x, ref_y, 'k:', label='y=-x')

    # Ejes y formato
    plt.xlabel(r'$(\Delta \rho)^2 + (\Delta \theta)^2$')
    plt.ylabel(r'E')
    plt.title(r'E vs $(\Delta \rho)^2 + (\Delta \theta)^2$')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Guardar figura

    ruta = ruta_output + "comparar_GS_SOR.png"
    plt.savefig(ruta, dpi=300)
    plt.show()

    print(f"\n📈 Pendiente ajustada (log–log, Gauss–Seidel): {pendiente:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Resolver Laplace en semidisco (modo simple o varios mallados)")
    parser.add_argument("--N_i", type=int, default=3, help="Número de intervalos radiales")
    parser.add_argument("--N_j", type=int, default=6, help="Número de intervalos angulares")
    parser.add_argument("--T0", type=float, default=-10, help="Temperatura interna")
    parser.add_argument("--T1", type=float, default=90, help="Temperatura externa")
    parser.add_argument("--modo", choices=["simple", "varios_mallados"], required=True, help="Modo de ejecución")
    parser.add_argument("--ruta_output", required=True, help="Ruta output")

    args = parser.parse_args()

    if args.modo == "simple":
        modo_simple(args.N_i, args.N_j, args.T0, args.T1, args.ruta_output)
    elif args.modo == "varios_mallados":
        varios_mallados(args.ruta_output)


if __name__ == "__main__":
    main()