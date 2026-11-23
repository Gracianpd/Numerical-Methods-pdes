import numpy as np
import matplotlib.pyplot as plt

def plot_stability_region(
    D, C,
    h_min=1e-3, h_max=None,
    k_min=0.0, k_max=2.0,
    n=600
):

    if C <= 0:
        raise ValueError("This plot assumes C > 0 (as in the derived region with h_max = sqrt(4D/C)).")

    h_crit = np.sqrt(4 * D / C)
    if h_max is None:
        h_max = 1.2 * h_crit

    # Build grid
    h = np.linspace(h_min, h_max, n)
    k = np.linspace(k_min, k_max, n)
    H, K = np.meshgrid(h, k)

    # Evaluate inequality safely
    expr = K * C - 4 * K * D / (H**2)

    stable = (expr >= -2) & (expr <= 0) & (K >= 0) & (H > 0)

    # Plot shaded region
    plt.figure(figsize=(7.5, 5.5))
    plt.pcolormesh(H, K, stable.astype(float), shading="auto", alpha=0.45)

    # Optional: plot the critical vertical line h = sqrt(4D/C)
    plt.axvline(h_crit, linestyle="--", linewidth=2, label=r"$h=2$")

    # Optional: plot the upper boundary where expr = -2 (when it exists)
    # From -2 = kC - 4kD/h^2  =>  k = 2 / (4D/h^2 - C)
    hh = np.linspace(h_min, min(h_max, 0.999*h_crit), 1200)
    denom = 4 * D / hh**2 - C
    k_upper = 2 / denom
    # keep only positive and within plot window
    valid = (k_upper > 0) & (k_upper <= k_max)
    plt.plot(hh[valid], k_upper[valid], linewidth=2, label=r"$k=\frac{2}{4/h^2 - 1}$")

    # Cosmetics
    plt.xlim(h_min, h_max)
    plt.ylim(k_min, k_max)
    plt.xlabel(r"$h$")
    plt.ylabel(r"$k$")
    plt.title(r"Región estable - paso temporal vs paso espacial")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_comparacion_soluciones(x, t, n_num, n_exa, tiempos_a_pintar, output_path, metodo):
    """
    Pinta numérica vs exacta para varios tiempos.
    """
    plt.figure()
    for time_target in tiempos_a_pintar:
        idx_t = np.argmin(np.abs(t - time_target))
        plt.plot(x, n_num[idx_t, :], "o-", label=f"Num t={t[idx_t]:.3f}")
        plt.plot(x, n_exa[idx_t, :], "--", label=f"Analítica t={t[idx_t]:.3f}")

    plt.xlabel("x")
    plt.ylabel("n(x,t)")
    plt.title(f"Densidad de neutrones numérica vs analítica. Método {metodo}")
    plt.legend()
    plt.grid(True)
    end = output_path + "comparacion_soluciones_" + metodo + ".png"
    plt.savefig(end)
    plt.show()

def plot_convergencia_espacial_log(h_arr, errL2_arr, errInf_arr):
    """
    Gráfico log-log de error vs h, con líneas de referencia h^1 y h^2.
    """
    plt.figure()
    plt.loglog(h_arr, errL2_arr, "o-", label="||e||_2")
    plt.loglog(h_arr, errInf_arr, "s--", label="||e||_∞")

    # Rectas de referencia de pendiente 1 y 2
    h_ref = np.array([h_arr.min(), h_arr.max()])
    E0 = errL2_arr[0]  # error L2 del h más grande, como referencia

    ref_p1 = E0 * (h_ref / h_ref[0])      # ~ h^1
    ref_p2 = E0 * (h_ref / h_ref[0])**2   # ~ h^2

    plt.loglog(h_ref, ref_p1, "k-.", label="ref ~ h^1")
    plt.loglog(h_ref, ref_p2, "k:",  label="ref ~ h^2")

    plt.gca().invert_xaxis()   # opcional: h grande a la izquierda
    plt.xlabel("h = Δx")
    plt.ylabel("error")
    plt.title("Convergencia espacial (error vs h)")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

def plot_densidad_promedio(t, n_prom_num, output_path):
    """
    Representa la densidad promedio de neutrones n(t) numérica y exacta.

    Parámetros
    ----------
    t : array (Nt+1,)
        Instantes temporales.
    n_prom_num : array (Nt+1,)
        n(t) calculada a partir de la solución numérica.
    n_prom_exa : array (Nt+1,)
        n(t) calculada a partir de la solución exacta.
    output_path : str
        Carpeta donde guardar la figura.
    """
    plt.figure()
    plt.plot(t, n_prom_num, "-", linewidth=1.0)

    plt.xlabel("t")
    plt.ylabel(r"Densidad promedio")
    plt.title("Evolución temporal de la densidad promedio de neutrones")
    plt.grid(True)
    plt.legend()

    fname = output_path + "densidad_promedio_t.png"
    plt.savefig(fname)
    plt.show()

def plot_densidad_promedio_multi(t, n_prom_dict, output_path):
    """
    Pinta la densidad promedio n(t) para varios valores de C.

    Parameters
    ----------
    t : array (Nt+1,)
        Tiempos.
    n_prom_dict : dict
        Diccionario {C: n_t} donde n_t tiene shape (Nt+1,).
    output_path : str
        Carpeta donde guardar la figura.
    """
    plt.figure()

    for C_val, n_t in n_prom_dict.items():
        plt.plot(t, n_t, linewidth=1.0, label=fr"$C={C_val:.2f}$")

    plt.xlabel("t")
    plt.ylabel(r"Densidad promedio")
    plt.title("Evolución temporal de la densidad promedio para distintos $C$")
    plt.grid(True)
    plt.legend()

    fname = output_path + "densidad_promedio_multi_C.png"
    plt.savefig(fname)
    plt.show()

def plot_convergencia_espacial_dos_metodos(h_prog, errL2_prog,
                                           h_cn,   errL2_cn):
    """
    Gráfico log-log de convergencia espacial comparando
    metodo progresivo y CN usando solo la norma L2.
    """
    plt.figure()

    # PROGRESIVO
    plt.loglog(h_prog, errL2_prog, "o-",  label="Prog  ||e||_2")

    # CRANK–NICOLSON
    plt.loglog(h_cn,   errL2_cn,   "s-",  label="CN   ||e||_2")

    # Referencia h^2 (usamos el h mayor de todos)
    h_ref_min = min(h_prog.min(), h_cn.min())
    h_ref_max = max(h_prog.max(), h_cn.max())
    h_ref = np.array([h_ref_min, h_ref_max])

    E0 = errL2_prog[0]  # escala de referencia
    ref_p2 = E0 * (h_ref / h_ref[0])**2
    plt.loglog(h_ref, ref_p2, "k-.", label="ref ~ h^2")

    plt.gca().invert_xaxis()
    plt.xlabel("h")
    plt.ylabel("Error")
    plt.title("Convergencia espacial (Prog vs CN)")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

def plot_convergencia_tiempo_dos_metodos(dt_prog, errL2_prog,
                                         dt_cn,   errL2_cn):
    """
    Gráfico log-log de convergencia en tiempo comparando
    metodo progresivo y CN (solo norma L2).
    """
    plt.figure()

    # PROGRESIVO
    plt.loglog(dt_prog, errL2_prog, "o-", label="Prog  ||e||_2")

    # CRANK–NICOLSON
    plt.loglog(dt_cn,   errL2_cn,   "s-", label="CN   ||e||_2")

    # Referencias ~ k^1 y ~ k^2
    dt_all = np.concatenate([dt_prog, dt_cn])
    dt_ref = np.array([dt_all.min(), dt_all.max()])

    E0 = errL2_prog[0]  # escala de referencia
    ref_k1 = E0 * (dt_ref / dt_ref[0])       # ~ k^1
    ref_k2 = E0 * (dt_ref / dt_ref[0])**2    # ~ k^2

    plt.loglog(dt_ref, ref_k1, "k-.", label="ref ~ k^1")
    plt.loglog(dt_ref, ref_k2, "k:",  label="ref ~ k^2")

    plt.gca().invert_xaxis()
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.title("Convergencia temporal (Prog vs CN)")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()
