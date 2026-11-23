import csv

def exportar_temperaturas_excel(T_GS, T_SOR, T_ex, T_cart, filename):
    #pone en un excel valores de temperaturas dimensionales
    #entrada: temperaturas y fichero
    #salidas: csv
    # Comprobamos longitud consistente

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Cabeceras
        writer.writerow(["T_GS", "T_SOR", "T_ex", "T_cart"])
        T_cart = T_cart[0]
        # Filas de datos
        for a, b, c, d in zip(T_GS, T_SOR, T_ex, T_cart):
            writer.writerow([f"{float(a):.6f}", f"{float(b):.6f}", f"{float(c):.6f}", f"{float(d):.6f}"])

    print(f"✅ Archivo '{filename}' generado correctamente.")

def exportar_adimensional_excel(u_GS, it_GS, u_SOR, it_SOR, u_ex, u_cart, ruta_csv):
    #pone en un excel valores de temperaturas adimensionales
    #entrada: temperaturas y fichero
    #salidas: csv
    # Longitud máxima para no perder datos si difieren
    n_max = max(len(u_GS) if u_GS is not None else 0,
                len(u_SOR) if u_SOR is not None else 0,
                len(u_ex)  if u_ex  is not None else 0,
                len(u_cart) if u_cart is not None else 0)

    def val(v, i):
        if v is None or i >= len(v):
            return ""
        return v[i]

    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Índice", "u_GS", "u_SOR", "u_ex", "u_cart"])

        for i in range(n_max):
            w.writerow([i, val(u_GS,i), val(u_SOR,i), val(u_ex,i), val(u_cart,i)])

        # bloque de iteraciones
        w.writerow([])
        if it_GS is not None:
            w.writerow(["Iteraciones_GS", it_GS])
        if it_SOR is not None:
            w.writerow(["Iteraciones_SOR", it_SOR])


    print(f"✅ Soluciones e iteraciones exportadas en '{ruta_csv}'")