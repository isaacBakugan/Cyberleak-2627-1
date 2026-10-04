# Nombre del integrante: Richard Yrady
# Cédula del integrante: 30715365

# haga su tarea aqui

import csv
import math
import matplotlib.pyplot as plt

 

def escalon(z):
    return 1 if z >= 0 else 0


def sigmoide(z):
    s = 1 / (1 + math.exp(-z))
    return 1 if s >= 0.5 else 0


activaciones = {
    "1": ("Escalon", escalon),
    "2": ("Sigmoide", sigmoide),
}

def cargar_csv(ruta):
    filas = []
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        lector = csv.reader(f)
        encabezado = next(lector, None)
        for fila in lector:
            if not fila:
                continue
            valores = [float(v) for v in fila]
            filas.append(valores)
    entradas = [f[:-1] for f in filas]
    salidas = [f[-1] for f in filas]
    return entradas, salidas



def predecir(entradas, pesos, sesgo, activacion):
    predicciones = []
    for x in entradas:
        z = sesgo + sum(w * xi for w, xi in zip(pesos, x))
        predicciones.append(activacion(z))
    return predicciones


def pedir_float(mensaje):
    while True:
        try:
            return float(input(mensaje))
        except ValueError:
            print("Por favor ingrese un numero valido.")


def pedir_pesos(n_entradas):
    sesgo = pedir_float("  Peso del sesgo (bias): ")
    pesos = []
    for i in range(n_entradas):
        pesos.append(pedir_float(f"  Peso para la columna x{i+1}: "))
    return sesgo, pesos


def pedir_activacion():
    print("  Funciones de activacion disponibles:")
    for clave, (nombre, _) in activaciones.items():
        print(f"    {clave}) {nombre}")
    while True:
        opcion = input("  Elija una funcion de activacion (1/2): ").strip()
        if opcion in activaciones:
            nombre, funcion = activaciones[opcion]
            return nombre, funcion
        print("  Opcion invalida.")


def graficar(entradas, esperado, predicho):
    x1 = [x[0] for x in entradas]
    x2 = [x[1] if len(x) > 1 else 0 for x in entradas]

    coincide = [e == p for e, p in zip(esperado, predicho)]
    colores_coincidencia = ["green" if c else "red" for c in coincide]

    fig, ejes = plt.subplots(1, 3, figsize=(15, 5))

    ejes[0].scatter(x1, x2, c=esperado, cmap="coolwarm")
    ejes[0].set_title("Valor esperado")

    ejes[1].scatter(x1, x2, c=predicho, cmap="coolwarm")
    ejes[1].set_title("Valor predicho")

    ejes[2].scatter(x1, x2, c=colores_coincidencia)
    ejes[2].set_title("Coincidencia (verde=si, rojo=no)")

    for eje in ejes:
        eje.set_xlabel("x1")
        eje.set_ylabel("x2")

    plt.tight_layout()
    plt.show()



def main():
    print("=== Perceptron ===")
    ruta = input("Ingrese la ruta del archivo CSV: ").strip()
    entradas, esperado = cargar_csv(ruta)
    n_entradas = len(entradas[0])
    print(f"Se cargaron {len(entradas)} vectores con {n_entradas} entrada(s) cada uno.\n")


    esperado_binario = [round(y) for y in esperado]

    while True:
        nombre_activacion, activacion = pedir_activacion()
        sesgo, pesos = pedir_pesos(n_entradas)

        predicho = predecir(entradas, pesos, sesgo, activacion)

        aciertos = sum(1 for e, p in zip(esperado_binario, predicho) if e == p)
        print(f"\nActivacion: {nombre_activacion} | Aciertos: {aciertos}/{len(entradas)}\n")

        graficar(entradas, esperado_binario, predicho)

        de_nuevo = input("Desea probar con otros pesos? (s/n): ").strip().lower()
        if de_nuevo != "s":
            print("Fin del programa.")
            break


if __name__ == "__main__":
    main()
