import numpy as np
import matplotlib.pyplot as plt
import sys


# ============================================================
# Exact solution (shared)
# ============================================================
def u_exact(x):
    # Exact solution for: u'' + u = -x, u(0)=0, u'(1)=0
    return (np.sin(x) / np.cos(1.0)) - x


# ============================================================
# P1 (Linear) FEM
# ============================================================
def fem_galerkin_1d_uniform(E=4):
    """
    Resuelve por Galerkin usando polinomios lineales (P1)
    entrada: numero de elementos finitos (uniformes)
    salida: nodos, valores
    """
    h = 1.0 / E
    nnode = E + 1

    x_nodes = np.arange(nnode, dtype=float) * h

    A = np.zeros((nnode, nnode), dtype=float)
    b = np.zeros(nnode, dtype=float)

    Ke = (1.0 / h) * np.array([[1.0, -1.0],
                               [-1.0, 1.0]])
    Me = (h / 6.0) * np.array([[2.0, 1.0],
                               [1.0, 2.0]])
    Ae = Ke - Me

    for e in range(E):
        xL = e * h
        xR = (e + 1) * h

        be = (h / 6.0) * np.array([2.0 * xL + xR,
                                   xL + 2.0 * xR], dtype=float)

        idx = [e, e + 1]
        A[np.ix_(idx, idx)] += Ae
        b[idx] += be

    # CC
    u0 = 0.0  # Dirichlet u(0)=0
    A_red = A[1:, 1:]
    b_red = b[1:] - A[1:, 0] * u0
    # Neumann u'(1)=0 es natural y homogénea -> nada que añadir

    u = np.zeros(nnode, dtype=float)
    u[0] = u0
    u[1:] = np.linalg.solve(A_red, b_red)

    return x_nodes, u


def plot_solution_P1(E, x_nodes, u_fem):
    x_fine = np.linspace(0.0, 1.0, 2001)
    u_fine = u_exact(x_fine)

    plt.figure()
    plt.plot(x_fine, u_fine, label="Exacta")
    plt.plot(x_nodes, u_fem, marker="o", label=f"FEM P1 (E={E})")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Solución exacta vs FEM (P1)")
    plt.legend()
    plt.grid(True)


def convergence_study_P1(E_list):
    hs = []
    emax = []

    for E in E_list:
        h = 1.0 / E
        x_nodes, u_fem = fem_galerkin_1d_uniform(E=E)
        u_ex = u_exact(x_nodes)
        err = u_fem - u_ex

        hs.append(h)
        emax.append(np.max(np.abs(err)))

    hs = np.array(hs, dtype=float)
    emax = np.array(emax, dtype=float)

    logh = np.log(hs)
    loge = np.log(emax)
    p, c = np.polyfit(logh, loge, 1)

    plt.figure()
    plt.loglog(hs, emax, marker="o", label=r"$\max |e|$ (nodos)")
    plt.loglog(hs, np.exp(c) * hs**p, linestyle="--", label=f"Ajuste: pendiente={p:.3f}")
    plt.xlabel("h")
    plt.ylabel(r"$\max |u_{\mathrm{FEM}}-u_{\mathrm{exact}}|$")
    plt.title("Convergencia FEM (P1, escala logarítmica)")
    plt.legend()
    plt.grid(True, which="both")

    return hs, emax, p, c


def run_lineal():
    # Resultados para algunos E con tabla + gráficas
    E_to_print = [4, 8]

    for E in E_to_print:
        x_nodes, u_fem = fem_galerkin_1d_uniform(E=E)
        u_ex = u_exact(x_nodes)
        diff = u_fem - u_ex
        abs_diff = np.abs(diff)

        print(f"FEM Galerkin 1D (uniform) with E={E} linear elements")
        print("Node    x_i        u_FEM         u_exact       (u_FEM-u_exact)   |error|")
        for i, (xi, uf, ue, d, ad) in enumerate(zip(x_nodes, u_fem, u_ex, diff, abs_diff)):
            print(f"{i:>3d}  {xi:8.5f}  {uf:12.8f}  {ue:12.8f}  {d:14.8e}  {ad:10.3e}")

        print("\nMax |error| at nodes =", np.max(abs_diff))
        print("-" * 70)

        plot_solution_P1(E, x_nodes, u_fem)

    # Convergence study
    E_list = [4, 8, 16, 32, 64, 128]
    hs, emax, slope, intercept = convergence_study_P1(E_list)

    print("\nConvergence study (P1, max nodal error):")
    for h, em in zip(hs, emax):
        print(f"h={h:10.6e}  max|error|={em:12.6e}")
    print(f"\nAjuste log-log: log(err) = p*log(h) + c  ->  p = {slope:.6f}, c = {intercept:.6f}")
    print("Orden observado ~ p (esperable ~2 en nodos para P1 en problemas suaves).")

    plt.show()


# ============================================================
# P2 (Quadratic) FEM
# ============================================================
def fem_galerkin_1d_quadratic_P2_uniform(E=2):
    """
    Resuelve por Galerkin usando polinomios cuadraticos (P2)
    entrada: numero de elementos finitos (uniformes)
    salida: nodos, valores
    """
    h = 1.0 / E
    nnode = 2 * E + 1

    # Nodos globales
    x_nodes = np.arange(nnode, dtype=float) * (h / 2.0)

    # Nodos
    conn = [[2 * e, 2 * e + 1, 2 * e + 2] for e in range(E)]

    A = np.zeros((nnode, nnode), dtype=float)
    b = np.zeros(nnode, dtype=float)

    #  matrices ref
    K_ref = np.array([
        [7 / 3, -8 / 3, 1 / 3],
        [-8 / 3, 16 / 3, -8 / 3],
        [1 / 3, -8 / 3, 7 / 3]
    ], dtype=float)

    M_ref = np.array([
        [2 / 15, 1 / 15, -1 / 30],
        [1 / 15, 8 / 15, 1 / 15],
        [-1 / 30, 1 / 15, 2 / 15]
    ], dtype=float)

    #  f(x)=x
    I0 = np.array([1 / 6, 2 / 3, 1 / 6], dtype=float)  # ∫_0^1 N(s) ds
    I1 = np.array([0.0, 1 / 3, 1 / 6], dtype=float)    # ∫_0^1 s N(s) ds

    Ke = (1.0 / h) * K_ref
    Me = h * M_ref
    Ae = Ke - Me

    for e, nodes in enumerate(conn):
        xL = e * h
        be = h * (xL * I0 + h * I1)

        A[np.ix_(nodes, nodes)] += Ae
        b[nodes] += be

    # Dirichlet at node 0
    u = np.zeros(nnode, dtype=float)
    u0 = 0.0
    u[0] = u0

    A_red = A[1:, 1:]
    b_red = b[1:] - A[1:, 0] * u0

    # Neumann u'(1)=0
    u[1:] = np.linalg.solve(A_red, b_red)

    return x_nodes, u


def fem_eval_P2_uniform(xq, u_nodes, E):
    """
    Evaluate P2 FEM solution u_h(x) at query points xq in [0,1]
    for uniform mesh with E elements.
    """
    xq = np.asarray(xq, dtype=float)
    h = 1.0 / E

    e = np.floor(xq / h).astype(int)
    e = np.clip(e, 0, E - 1)

    xL = e * h
    s = (xq - xL) / h

    N1 = 2.0 * (s - 0.5) * (s - 1.0)
    N2 = 4.0 * s * (1.0 - s)
    N3 = 2.0 * s * (s - 0.5)

    i0 = 2 * e
    i1 = 2 * e + 1
    i2 = 2 * e + 2

    return N1 * u_nodes[i0] + N2 * u_nodes[i1] + N3 * u_nodes[i2]


def plot_exact_vs_fem_P2(E, x_nodes, u_nodes):
    x_fine = np.linspace(0.0, 1.0, 2001)
    u_fine = u_exact(x_fine)
    u_fem_fine = fem_eval_P2_uniform(x_fine, u_nodes, E)

    plt.figure()
    plt.plot(x_fine, u_fine, label="Exacta")
    plt.plot(x_fine, u_fem_fine, label=f"FEM (E={E})")
    plt.plot(x_nodes, u_nodes, marker="o", linestyle="None", label="Nodos FEM")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Solución exacta vs FEM (P2)")
    plt.legend()
    plt.grid(True)


def convergence_study_P2(E_list):
    hs = []
    emax = []

    for E in E_list:
        h = 1.0 / E
        x_nodes, u_fem = fem_galerkin_1d_quadratic_P2_uniform(E=E)
        u_ex = u_exact(x_nodes)
        err = u_fem - u_ex

        hs.append(h)
        emax.append(np.max(np.abs(err)))

    hs = np.array(hs, dtype=float)
    emax = np.array(emax, dtype=float)

    logh = np.log(hs)
    loge = np.log(emax)
    p, c = np.polyfit(logh, loge, 1)

    plt.figure()
    plt.loglog(hs, emax, marker="o", label=r"$\max |e|$ (en nodos)")
    plt.loglog(hs, np.exp(c) * hs**p, linestyle="--", label=f"Ajuste: pendiente={p:.3f}")
    plt.xlabel("h")
    plt.ylabel(r"$\max |u_{\mathrm{FEM}}-u_{\mathrm{exact}}|$")
    plt.title("Convergencia FEM (P2, escala logarítmica)")
    plt.legend()
    plt.grid(True, which="both")

    return hs, emax, p, c


def run_cuadratico():
    # Tablas + gráficas para E=2 y E=4
    for E in [2, 4]:
        x_nodes, u_fem = fem_galerkin_1d_quadratic_P2_uniform(E=E)
        u_ex = u_exact(x_nodes)
        diff = u_fem - u_ex
        abs_diff = np.abs(diff)

        print(f"FEM Galerkin 1D with quadratic (P2) elements, E={E} (uniform, no xi)")
        print("Node    x_i        u_FEM         u_exact       (u_FEM - u_exact)    |error|")
        for i, (xi, uf, ue, d, ad) in enumerate(zip(x_nodes, u_fem, u_ex, diff, abs_diff)):
            print(f"{i:>3d}  {xi:8.5f}  {uf:12.8f}  {ue:12.8f}  {d:14.8e}  {ad:10.3e}")

        print("\nMax |error| at nodes =", np.max(abs_diff))
        print("-" * 75)

        plot_exact_vs_fem_P2(E, x_nodes, u_fem)

    # Estudio de convergencia
    E_list = [2, 4, 8, 16, 32, 64, 128]
    hs, emax, slope, intercept = convergence_study_P2(E_list)

    print("\nConvergence study (P2, max nodal error):")
    for h, em in zip(hs, emax):
        print(f"h={h:10.6e}  max|error|={em:12.6e}")
    print(f"\nAjuste log-log: log(err) = p*log(h) + c  ->  p = {slope:.6f}, c = {intercept:.6f}")

    plt.show()


# ============================================================
# CLI
# ============================================================
def usage():
    print("Uso:")
    print("  python fem_1d.py lineal")
    print("  python fem_1d.py cuadratico")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    modo = sys.argv[1].strip().lower()

    if modo == "lineal":
        run_lineal()
    elif modo == "cuadratico":
        run_cuadratico()
    else:
        print(f"Modo no reconocido: {modo}")
        usage()
        sys.exit(1)
