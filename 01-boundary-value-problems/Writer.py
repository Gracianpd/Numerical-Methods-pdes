


import csv


def guardar_vector_csv(vector, nombre_archivo="vector.csv"):
    # crea fichero csv con resultados xi, wi
    with open(nombre_archivo, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "w"])  # encabezado

        for elemento in vector:
            if isinstance(elemento, (list, tuple)):
                # Si es una lista de tuplas, iteramos sobre ella
                if isinstance(elemento, list):
                    for x, w_val in elemento:
                        writer.writerow([x, w_val])
                else:  # si es una tupla simple
                    writer.writerow(elemento)
            else:
                writer.writerow([elemento])

    print(f"✅ Vector guardado en '{nombre_archivo}'")

def guardar_en_csv(diccionario, nombre_archivo):
    # crea fichero csv con resultados xi, w1i y w2i

    with open(nombre_archivo, mode='w', newline='') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(['x', 'w1', 'w2'])  # cabecera

        for x, valores in diccionario.items():
            w1, w2 = valores  # desempacamos la tupla
            escritor.writerow([x, w1, w2])

    print(f"Archivo '{nombre_archivo}' creado correctamente.")