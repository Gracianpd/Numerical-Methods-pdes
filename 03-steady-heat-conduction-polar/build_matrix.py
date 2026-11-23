import numpy as np


def build_matriz(dtheta, i_min, i_max, j_min, j_max):
    #consrtuye matriz a a partie de pasos y tamaño
    #entradas: paso de theta, valor de radaio minimo y maximo, valor de angulo minimo y maximo
    #salida: matriz A y vector b
    Ni = i_max - i_min + 1
    Nj = j_max - j_min + 1
    N = Ni * Nj

    A = np.zeros((N, N), dtype=float)
    b = np.zeros(N, dtype=float)

    def local_idx(i, j):
        """Índice lineal (i,j) → k."""
        return (i - i_min) * Nj + (j - j_min)

    for i in range(i_min, i_max + 1):
        arp = 1.0 + 1.0 / (2.0 * i)                   # a_{r+}
        arm = 1.0 - 1.0 / (2.0 * i)                   # a_{r-}
        ath = 1.0 / (i * i * (dtheta * dtheta))       # a_θ
        aP  = 2.0 * (1.0 + 1.0 / (i * i * (dtheta * dtheta)))

        for j in range(j_min, j_max + 1):
            k = local_idx(i, j)
            A[k, k] = aP

            # --- Vecino radial i+1 (hacia el arco)
            if i + 1 <= i_max:
                A[k, local_idx(i + 1, j)] -= arp
            elif i + 1 == i_max + 1:
                # Borde exterior:
                b[k] += arp

            # --- Vecino radial i-1 (hacia el centro)
            if i - 1 >= i_min:
                A[k, local_idx(i - 1, j)] -= arm
            # si i-1 < i_min  omitido (ρ=0, no incluido)

            # --- Vecino angular j+1
            if j + 1 <= j_max:
                A[k, local_idx(i, j + 1)] -= ath
            elif j + 1 == j_max + 1:
                # Diámetro superior  u=0, no se añade nada
                pass

            # --- Vecino angular j-1
            if j - 1 >= j_min:
                A[k, local_idx(i, j - 1)] -= ath
            elif j - 1 == j_min - 1:
                # Diámetro inferior  u=0, no se añade nada
                pass

    return A, b


def build_matriz_cartesian():
    #contructor de matriz cartesiana. Hardcoded
    A = np.array([
        [4, -1, 0, 0, 0, 0, 0, 0],
        [-1, 4, -1, 0, 0, -1, 0, 0],
        [0, -1, 4, -1, 0, 0, -1, 0],
        [0, 0, -1, 4, -1, 0, 0, -1],
        [0, 0, 0, -1, 4, 0, 0, 0],
        [0, -1, 0, 0, 0, 4, -1, 0],
        [0, 0, -1, 0, 0, -1, 4, -1],
        [0, 0, 0, -1, 0, 0, -1, 4],
    ], dtype=float)

    b = np.array([2,0,0,0,2,2,1,1])
    return A, b