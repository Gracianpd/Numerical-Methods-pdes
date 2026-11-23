###ESTE MODULO RESUELVE UNA EDO LINEAL CON PROBLEMA DE VALOR DE FRONTERA###
#Sus entradas deben ser los extremos a,b y sus respectivos valores alpha, beta. Además se debe conocer
#la ecuación
import numpy as np

def valor_n_1(a,b, alpha, N):
    #esta funcion calcula los valores de N=1 de la matriz tridiagonal
    #entradas: extremos del problema: a, b; valor inicial en a: alpha; numero entero en el que dividiremos en N + 1 subintervalos iguales el intervalo considerado: N
    #salidas: paso: h; valores de la primera fila de la matriz tridiagonal a resolver: a1, b1; valor del vector resultante de la ecucacion; d1
    h = (b - a) / (N + 1)
    x = a + h
    px = 0
    qx = -(1 + (x - 1) ** 2)
    rx = -1

    a1 = 2 + h ** 2 * qx
    b1 = -1 + (h/2) * px
    d1 = (-h ** 2 * rx) + (1 + (h/2)*px) * alpha
    return h, a1, b1, d1

def valor_i(a , b, N, h):
    #esta funcion calcula los valores de N=2,...,N-1 de la matriz tridiagonal
    #entradas: a, b, N, h descritas en funcion anterior
    #salidas: diccionario con key: valor de x en el que se estan calculando los valores de la matrix tridiagonal y el vector resultante: di
    value_dict = {}
    for i in range(2, N):
        x = a + i * h
        px = 0
        qx = -(1+ (x - 1) ** 2)
        rx = -1

        ai = 2 + h ** 2 * qx
        bi = -1 + (h / 2) * px
        ci = -1 - (h / 2) * px
        di = (-h ** 2 * rx)
        value_dict[i] = (ai, bi, ci, di)
    return value_dict

def valor_n_ultimo(b, h, beta):
    #esta funcion calcula los valores N de la matriz tridiagonal
    #entradas: b, h explicadas en funciones anteriores; beta: valor de la funcion en el punto b
    #salidas: valores ultimos de la matrix tridiagonal: an, cn; y valor ultimo del vector resultante: dn
    x = b - h
    px = 0
    qx = -(1 + (x - 1) ** 2)
    rx = -1

    an = 2 + h ** 2 * qx
    cn = -1 - (h / 2) * px
    dn = (-h ** 2 * rx) + (1 - (h/2) * px) * beta
    return an, cn, dn

def resolver_tridiagonal_1(a1, b1, d1):
    #esta funcion calcula los primeros valores de solución del sistema tridiagonal
    #entradas: salidas de funcion "valor_n_1", explicadas ahi
    #salidas: primeros coeficientes para la resolucion del sistema tridiagonal: l1, u1, z1
    l1 = a1
    u1 = b1 / a1
    z1 = d1 / l1
    return l1, u1, z1

def resolver_tridiagonal_i(value_dict, u1, z1):
    #esta funcion calcula los primeros valores de N=2,...,N-1 de la solución del sistema tridiagonal
    #entradas: salidas de funcion "valor_i", explicadas ahi: value_dict; salidas de la funcion "resolver_tridiagonal_1", explicadas ahi: u1, z1
    #salidas: diccionario de coeficientes para la resolucion del sistema tridiagonal hasta n - 1, con diccionario con key: valor x, value: ui, zi;
    #penultimos valores de u y z por comodidad de programacion
    tri_sol_dict = {}
    u_prev = u1
    z_prev = z1
    for key, value in value_dict.items():
        ai, bi, ci, di = value
        li = ai - ci * u_prev
        ui = bi / li
        zi = (di - ci * z_prev) / li
        tri_sol_dict[key] = (ui, zi)
        u_prev = ui
        z_prev = zi
    return tri_sol_dict, u_prev, z_prev

def resolver_tridiagonal_ultimo_n(an, cn, dn, u_prev, z_prev):
    #esta funcion calcula los valores N de la solución del sistema tridiagonal
    #entradas: salidas de funcion "valor_n_ultimo", explicadas ahi; ultimos valores de u y z
    #salidas: ultimos coeficientes para la resolucion del sistema tridiagonal: ln, zn
    ln = an - cn * u_prev
    zn = (dn - cn * z_prev) / ln
    return ln, zn

def crear_vect_sol(a, b, alpha, beta, u1, z1, zn, h, tri_sol_dict):
    #crea el vector de soluciones w
    #entradas: todas explicadas en funciones anteriores. Solo cabe recordar que la solucion aproximada del problema se construye a partir de los coeficientes
    #de resolucion de la matriz tridiagonal
    #salidas: vector w con dupla: punto de la malla, valor de la solucion aproximada en ese punto
    w = []
    intermediate_w = []
    w_n_plus_1 = beta
    wn = zn
    w_next = wn #definicion redundante por claridad de código
    for key, value in reversed(tri_sol_dict.items()):
        #importante: este for funciona porque nuestro diccionario siempre tendra un tamaño que recorrerá de
        # i = N - 1 a i = 2. Nótese que los valores i = 0 e i = N + 1 están definidas por CC
        ui, zi = value
        wi = zi - ui * w_next
        x = a + key * h
        intermediate_w.append((x, wi))
        w_next = wi
    w.append((a, alpha))
    #para i = 1
    w1 = z1 - u1 * w_next
    w.append((a + h, w1))
    #para i = 2, ... , N - 1
    w.append(list(reversed(intermediate_w)))
    #para i = N
    w.append((b - h, wn))
    #para i = N + 1
    w.append((b, beta))
    return w







