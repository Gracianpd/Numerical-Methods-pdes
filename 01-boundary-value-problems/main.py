import os
import argparse
from DiferenciasFinitasLineales import *
from DisparoLineal import *
from Writer import *
from Plotter import *

def main():
    parser = argparse.ArgumentParser(description="Resolver EDO usando Diferencias Finitas o Disparo Lineal")
    parser.add_argument("--a", type=float, required=True, help="Extremo izquierdo")
    parser.add_argument("--b", type=float, required=True, help="Extremo derecho")
    parser.add_argument("--alpha", type=float, required=True, help="Condición de frontera en x=a")
    parser.add_argument("--beta", type=float, required=True, help="Condición de frontera en x=b")
    parser.add_argument("--N", type=int, nargs="+", required=True,
                        help="Número(s) de subdivisiones, primer valor para disparo segundo para diferencias finitas.")
    parser.add_argument("--modo", type=str, required=True, choices=["diferencias", "disparo", "dos_metodos"], help="Método numérico a usar")
    parser.add_argument("--output", type=str, required=True, help="Directorio de salida de csv e imagenes.")

    args = parser.parse_args()
    a, b, alpha, beta, N , modo, output= args.a, args.b, args.alpha, args.beta, args.N, args.modo, args.output
    # Verificación según modo
    if modo == "dos_metodos":
        if len(args.N) != 2:
            parser.error("Para modo 'dos_metodos', debe especificar dos valores en --N (por ejemplo: --N 1000 2000)")
        N1, N2 = args.N
    else:
        if len(args.N) != 1:
            parser.error(f"Para modo '{modo}', debe especificar un único valor en --N")
        N1 = args.N[0]
        N2 = None

    if modo == "diferencias":
        #DIFERENCIAS FINITAS
        #obtener coeficientes de matriz
        h, a1, b1, d1 = valor_n_1(a,b, alpha, N1)
        dict_ai = valor_i(a , b, N1, h)
        an, cn, dn = valor_n_ultimo(b, h, beta)

        #resolver matriz tridiagonal
        l1, u1, z1 = resolver_tridiagonal_1(a1, b1, d1)
        tri_sol_dict, u_prev, z_prev = resolver_tridiagonal_i(dict_ai, u1, z1)
        ln, zn = resolver_tridiagonal_ultimo_n(an, cn, dn, u_prev, z_prev)

        #asignar soluciones wi
        w = crear_vect_sol(a, b, alpha, beta, u1, z1, zn, h, tri_sol_dict)
        guardar_vector_csv(w, os.path.join(output, "resultados_diferencias_finitas.csv"))
        salida_imagen = os.path.join(output, "grafica_dl.pdf")
        plot_w_diferencias(w,savepath=salida_imagen)

    if modo == "disparo":
        #DISPARO LINEAL
        #buscamos resolver ecuaciones de condicion inicial por metodo rugne-kutta
        h, u10, u20, v10, v20 = coef_iniciales(a, b, alpha, N1)
        u_values, v_values = rugne_kuta(a, h, N1, u10, u20, v10, v20)

        #calcular aproximacion a solucion en cada punto a partir de soluciones de ecuaciones con condicion inicial
        w = calcular_w(u_values, v_values, alpha, beta, a, h)
        guardar_en_csv(w, os.path.join(output, "resultados_disparo.csv"))
        salida_imagen = os.path.join(output, "grafica_disparo.pdf")
        plot_w1_disparo(w, savepath=salida_imagen)

    if modo == "dos_metodos":
        #DIFERENCIAS FINITAS
        h, a1, b1, d1 = valor_n_1(a,b, alpha, N2)
        dict_ai = valor_i(a , b, N2, h)
        an, cn, dn = valor_n_ultimo(b, h, beta)

        #resolver matriz tridiagonal
        l1, u1, z1 = resolver_tridiagonal_1(a1, b1, d1)
        tri_sol_dict, u_prev, z_prev = resolver_tridiagonal_i(dict_ai, u1, z1)
        ln, zn = resolver_tridiagonal_ultimo_n(an, cn, dn, u_prev, z_prev)

        #asignar soluciones wi
        w_df = crear_vect_sol(a, b, alpha, beta, u1, z1, zn, h, tri_sol_dict)
        guardar_vector_csv(w_df, os.path.join(output, "resultados_diferencias_finitas.csv"))

        #DISPARO LINEAL
        #buscamos resolver ecuaciones de condicion inicial por metodo rugne-kutta
        h, u10, u20, v10, v20 = coef_iniciales(a, b, alpha, N1)
        u_values, v_values = rugne_kuta(a, h, N1, u10, u20, v10, v20)

        #calcular aproximacion a solucion en cada punto a partir de soluciones de ecuaciones con condicion inicial
        w_disparo = calcular_w(u_values, v_values, alpha, beta, a, h)
        guardar_en_csv(w_disparo, os.path.join(output, "resultados_disparo.csv"))
        salida_imagen = os.path.join(output, "grafica_superpuesta.pdf")
        plot_w_comparacion(w_disparo, w_df,savepath=salida_imagen)


if __name__ == "__main__":
    main()