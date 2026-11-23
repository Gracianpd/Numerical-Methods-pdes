import matplotlib.pyplot as plt

def plot_w1_disparo(w, title="Solución Referencia $w_1(x). Disparo Lineal, N=2000$", savepath=None):
    #esta funcion pinta w1 en funcion de x para el metodo del disparo
    x_vals = sorted(w.keys())
    w1_vals = [w[x][0] for x in x_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, w1_vals, marker='o', color='blue', linewidth=0.005, markersize=1)
    plt.xlabel("x")
    plt.ylabel(r"$w_1(x)$")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada en: {savepath}")

    plt.show()


def plot_w_diferencias(w, title="Solución Referencia $w(x). Diferencias Finitas, N=20099$", savepath=None):
    #estsa funcion pinta los wi en funcion de x para el metodo de las diferencias finitas

    flat_w = []
    for item in w:
        if isinstance(item, list):
            flat_w.extend(item)
        else:
            flat_w.append(item)

    w_sorted = sorted(flat_w, key=lambda t: t[0])
    x_vals = [x for x, _ in w_sorted]
    w_vals = [val for _, val in w_sorted]

    plt.figure(figsize=(8, 5))
    plt.plot(
        x_vals, w_vals,
        marker='o', color='blue',
        linewidth=1.5, markersize=1
    )
    plt.xlabel("x")
    plt.ylabel(r"$w(x)$")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada en: {savepath}")

    plt.show()


def plot_w_comparacion(w1, w2,
                       title="Comparación de soluciones $w(x)$",
                       label1="Disparo lineal",
                       label2="Diferencias finitas",
                       savepath=None):
    #esta funcion dibuja una grafica de dos soluciones por metodos diferentes superpuestas
    #sus entradas son, a parte de path de guardado y labels w1: solucion disparo, w2: solucion diferencias finitas

    def procesar_w(w):
        # Acepta tanto dicts {x: (w, ...)} como listas [(x, w)]
        if isinstance(w, dict):
            x_vals = sorted(w.keys())
            w_vals = [w[x][0] if isinstance(w[x], (list, tuple)) else w[x] for x in x_vals]
        else:
            flat_w = []
            for item in w:
                if isinstance(item, list):
                    flat_w.extend(item)
                else:
                    flat_w.append(item)
            w_sorted = sorted(flat_w, key=lambda t: t[0])
            x_vals = [x for x, _ in w_sorted]
            w_vals = [val for _, val in w_sorted]
        return x_vals, w_vals

    x1, y1 = procesar_w(w1)
    x2, y2 = procesar_w(w2)

    plt.figure(figsize=(8, 5))
    plt.plot(x1, y1, color='blue', linewidth=1.5, label=label1)
    step = max(1, len(x2) // 100)  # ajusta según cuántos puntos quieras ver
    plt.plot(x2[::step], y2[::step], 'o', color='red', markersize=2, label=label2, linestyle='None')

    plt.xlabel("x")
    plt.ylabel(r"$w(x)$")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada en: {savepath}")

    plt.show()
