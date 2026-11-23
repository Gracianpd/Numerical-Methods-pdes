import numpy as np

def coef_iniciales(a, b, alpha , N):
    #calcula los primeros coeficientes del metodo Rugne-Kutta
    # entradas: extremos del problema: a, b; valor inicial en a: alpha; numero de puntos considerado: N
    #salidas: paso: h; valor inicial de las aproximaciones a las soluciones de las ecuaciones de valor inicial y sus derivadas: u10, u20, v10, v20
    h = (b - a) / N
    u10 = alpha
    u20 = 0
    v10 = 0
    v20 = 1
    return h, u10, u20, v10, v20

def rugne_kuta(a, h, N, u10, u20, v10, v20):
    #calcula los ui, vi utilizados para obtener las soluciones a las ecuaciones de valor inicial W1, W2
    #entradas: explicadas en funcion anterior
    #salidas: arrays u_values, v_values conteniendo las soluciones daproximadas e las ecuaciones de valor inicial calculadas por metodo de rugne-kutta
    #asignamos primeros valores de u1i, u2i, v1i, v2i
    u2i = u20
    u1i = u10
    v1i = v10
    v2i = v20
    u_values = []
    v_values = []
    for i in range(0, N):
        #valores polinomios sin despalzar
        x = a + i * h
        px = 0
        qx = -(1+ (x - 1) ** 2)
        rx = -1

        #valores polinomios desplazados h/2
        x_desp = x + (h / 2)
        px_desp = 0
        qx_desp = -(1+ (x_desp - 1) ** 2)
        rx_desp = -1

        #valores polinomios desplazados h
        x_desp2 = x + h
        px_desp2 = 0
        qx_desp2 = -(1+ (x_desp2 - 1) ** 2)
        rx_desp2 = -1

        #valores k para construir coeficientes u y u
        k11 = h * u2i
        k12 = h * (px * u2i + qx * u1i + rx)
        k21 = h * (u2i + 0.5 * k12)
        k22 = h * (px_desp * (u2i + 0.5 * k12) + qx_desp * (u1i + 0.5 * k11) + rx_desp)
        k31 = h * (u2i + 0.5 * k22)
        k32 = h * (px_desp * (u2i + 0.5 * k22) + qx_desp * (u1i + 0.5 * k21) + rx_desp)
        k41 = h * (u2i + k32)
        k42 = h * (px_desp2 * (u2i + k32) + qx_desp2 * (u1i + k31) + rx_desp2)
        u1i_next = u1i + (1 / 6) * (k11 + 2 * k21 + 2 * k31 + k41)
        u2i_next = u2i + (1 / 6) * (k12 + 2 * k22 + 2 * k32 + k42)
        #en u_values guardamos los ui con i = 1,...,N (los que se usaran para calcular w)
        u_values.append((u1i_next, u2i_next))
        u1i = u1i_next
        u2i = u2i_next

        #valores k para construir coeficientes v y v. cambio notacion k' -> t
        t11 = h * v2i
        t12 = h * (px * v2i + qx * v1i)
        t21 = h * (v2i + 0.5 * t12)
        t22 = h * (px_desp * (v2i + 0.5 * t12) + qx_desp * (v1i + 0.5 *t11))
        t31 = h * (v2i + 0.5 * t22)
        t32 = h * (px_desp * (v2i + 0.5 * t22) + qx_desp * (v1i + 0.5 * t21))
        t41 = h * (v2i + t32)
        t42 = h * (px_desp2 * (v2i + t32) + qx_desp2 * (v1i + t31))
        v1i_next = v1i + (1 / 6) * (t11 + 2 * t21 + 2 * t31 + t41)
        v2i_next = v2i + (1 / 6) * (t12 + 2 * t22 + 2 * t32 + t42)
        #en v_values guardamos los vi con i = 1,...,N (los que se usaran para calcular w)
        v_values.append((v1i_next, v2i_next))
        v1i = v1i_next
        v2i = v2i_next

    return u_values, v_values

def calcular_w(u_values, v_values, alpha, beta, a, h):
    #calcula la solucion aproximada w a partir de las soluciones de las ecuaciones de condiciones iniciales obtenidas por r-k
    #entradas: explicadas en funciones anteriores
    #salidas: diccionario w con key: valor de x en la malla; W1: valor aproximado de la funcion en cada punto, W2: valor aproximado de la derivada de la funcion en cada punto
    w = {}
    w10 = alpha
    w20 = (beta - u_values[-1][0]) / v_values[-1][0]
    w[a] = (w10, w20)

    for i in range(len(u_values)):
        x = a + (i + 1) * h

        W1 = u_values[i][0] + w20 * v_values[i][0]
        W2 = u_values[i][1] + w20 * v_values[i][1]
        w[x] = (W1, W2)
    return w