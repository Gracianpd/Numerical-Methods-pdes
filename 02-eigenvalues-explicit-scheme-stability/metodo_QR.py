import math

'''Metodo QR con y sin SHIFT'''
def QR_tridiagonal_algorithm(a_init, b_init, TOL, M, shift):
    #metodo QR para una tridiagonal
    #inputs: diagonal a, subdiagonal superior e inferior, tolerancia y maximo numero de iteraciones

    # Copiamos porque vamos a modificar
    a = a_init.copy()
    b = b_init.copy()
    autoval = []
    SHIFT = 0.0
    k = 1

    while k <= M:
        n = len(a)
        if n == 0:
            break
        # PASO 3: check si la esquina inferior es aproximadamente el autoval |b_n| ≤ TOL
        # b[n] es b[n-2] en Python
        if abs(b[-1]) < TOL:
            lam = a[-1] + SHIFT
            autoval.append([lam,k]) # Guarda el autovalor
            a.pop()  #Elimina a_n (a[n-1] ahora)
            if b: b.pop() # Elimina b_n (b[n-2] ahora)
            continue

        # PASO 4: check si la esquina superior es aproximadamente el autoval |b_n| ≤ TOL
        if abs(b[0]) <= TOL:  # b_2
            lam = a[0] + SHIFT
            autoval.append([lam,k])  # Guarda el autovalor

            # Reducir
            a.pop(0)  # Eliminar a_1
            b.pop(0)  # Eliminar b_2
            continue

        # PASO 5: si n = 0 PARAR
        if n == 0:
            break

        # PASO 6: si n = 1 el autoval es a_1 + SHIFT
        if n == 1:
            lam = a[0] + SHIFT
            autoval.append([lam,k])  # Guarda el último autovalor y hemos acabato
            break

        # PASO 7: división interna - en principio no usado en nuestro problema, pero implementado
        # revisa b_3, b_4, ..., b_{n}
        # El algoritmo se detiene si se encuentra una división y se le pide al usuario
        # que resuelva cada subproblema por separado.
        for j in range(1, n - 1):  # revisa b_3, b_4, ..., b_{n-1} (índices Python 1 a n-3)
            if abs(b[j]) <= TOL:
                # Si se divide, hay que resolver dos matrices más pequeñas.
                print(f"La matriz se divide en el paso {j + 2}. Implementación requiere manejo de submatrices.")
                return autoval  # Salida para simplificar

        # PASO 8: calcular b, c, d (shift dinámico)
        b_k = -(a[n - 2] + a[n - 1])  # -(a(n-1)+a(n))
        c_k = a[n - 1] * a[n - 2] - b[n - 2] ** 2  # a(n)a(n-1) - b(n)^2
        d_k = math.sqrt(b_k ** 2 - 4 * c_k)  # sqrt(b^2 - 4c)

        # PASO 9: calcular mu1, mu2
        if b_k > 0:
            mu1 = -2 * c_k / (b_k + d_k)
            mu2 = -(b_k + d_k) / 2
        else:
            mu1 = (d_k - b_k) / 2
            mu2 = 2 * c_k / (d_k - b_k)

        # PASO 10: si n=2 → ya tenemos autoval 1 y 2
        if n == 2:
            lam1 = mu1 + SHIFT
            lam2 = mu2 + SHIFT
            autoval.extend([lam1, lam2])
            break

        # PASO 11: elegir shift
        if shift:

            if abs(mu1 - a[-1]) < abs(mu2 - a[-1]):
                sigma = mu1
            else:
                sigma = mu2
        else:
            sigma = 0

        # PASO 12: acumular SHIFT
        SHIFT += sigma

        # PASO 13: realizar cambio d_j = a_j - sigma
        d = [a_j - sigma for a_j in a]

        # PASOS 14-15: cálculo de R^k
        n = len(d)
        x = [0] * n
        y = [0] * n
        z = [0] * n
        c = [0] * n
        s = [0] * n
        q = [0] * n

        x[0] = d[0]
        y[0] = b[0]

        for j in range(1, n): #j=2,...,n
            z[j - 1] = math.sqrt(x[j - 1] ** 2 + b[j - 1] ** 2)
            c[j] = x[j - 1] / z[j - 1]
            s[j] = b[j - 1] / z[j - 1]

            q[j - 1] = c[j] * y[j - 1] + s[j] * d[j]
            x[j] = -s[j] * y[j - 1] + c[j] * d[j]

            if j < n - 1:
                y[j] = c[j] * b[j]

        z[n - 1] = x[n - 1]

        # PASOS 16-18: construir A^(k+1)
        a_new = [0] * n
        b_new = [0] * (n - 1)

        a_new[0] = s[1] * q[0] + c[1] * z[0]
        b_new[0] = s[1] * z[1]

        for j in range(1, n - 1):
            a_new[j] = s[j + 1] * q[j] + c[j] * c[j + 1] * z[j]
            b_new[j] = s[j + 1] * z[j + 1]

        a_new[-1] = c[-1] * z[-1]

        a = a_new
        b = b_new

        # PASO 19: incrementar k
        k += 1

    # PASO 20: max iteraciones excedido
    if len(autoval) < len(a_init):
        print("Advertencia: Número máximo de iteraciones excedido. No todos los autovalores encontrados.")
    return autoval