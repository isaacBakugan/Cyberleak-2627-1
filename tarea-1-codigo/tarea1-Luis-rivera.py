# Nombre del integrante: Luis Rivera   
# Cédula del integrante: 27773794

import matplotlib.pyplot as plt


def cargar_csv(ruta_archivo):
    """Carga los datos desde un archivo CSV utilizando solo las herramientas nativas de Python.
    
    Retorna los vectores de entrada y los valores esperados.
    """
    entradas = []
    esperados = []
    
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea:
                continue
            
            partes = linea.split(',')
            try:
                # Convertir cada elemento a tipo flotante
                valores = [float(p) for p in partes]
            except ValueError:
                # Omitir líneas con encabezados no numéricos
                continue
            
            # Las primeras n-1 columnas son entradas, la n-ésima es el resultado esperado
            entradas.append(valores[:-1])
            esperados.append(valores[-1])
            
    return entradas, esperados


def funcion_suma(vector_entrada, pesos, peso_sesgo):
    """Calcula la suma ponderada del perceptrón:
    z = w_0 * 1 + sum(w_i * x_i)
    """
    suma = peso_sesgo * 1.0  # El valor de la entrada del sesgo x_0 es 1.0
    for x_i, w_i in zip(vector_entrada, pesos):
        suma += x_i * w_i
    return suma


def activacion_escalon_unipolar(z):
    """Función de activación Escalón Unipolar: retorna 1 si z >= 0, de lo contrario 0."""
    return 1 if z >= 0 else 0


def activacion_escalon_bipolar(z):
    """Función de activación Escalón Bipolar / Signo: retorna 1 si z >= 0, de lo contrario -1."""
    return 1 if z >= 0 else -1


def evaluar_datos(entradas, esperados, pesos, peso_sesgo, funcion_act):
    """Calcula la predicción para cada vector y evalúa coincidencia."""
    predicciones = []
    colores = []
    aciertos = 0

    for x, y_esperado in zip(entradas, esperados):
        z = funcion_suma(x, pesos, peso_sesgo)
        y_predicho = funcion_act(z)
        predicciones.append(y_predicho)

        coincide = (y_predicho == y_esperado)
        if coincide:
            aciertos += 1
            colores.append('green')
        else:
            colores.append('red')

    return predicciones, colores, aciertos


def graficar_resultados(entradas, esperados, predicciones, colores):
    """Muestra tres gráficos de dispersión mediante matplotlib.pyplot.scatter.
    
    Si n > 3, toma únicamente las primeras dos dimensiones de entrada.
    """
    # Extraer primera y segunda dimensión del vector de entrada
    x1 = [v[0] for v in entradas]
    x2 = [v[1] if len(v) > 1 else 0 for v in entradas]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Gráfico 1: Valor Esperado
    sc1 = axes[0].scatter(x1, x2, c=esperados, cmap='bwr', edgecolors='k', s=80)
    axes[0].set_title("1. Valor Esperado")
    axes[0].set_xlabel("Entrada X1")
    axes[0].set_ylabel("Entrada X2" if len(entradas[0]) > 1 else "Cero")
    fig.colorbar(sc1, ax=axes[0])

    # Gráfico 2: Valor Predicho
    sc2 = axes[1].scatter(x1, x2, c=predicciones, cmap='bwr', edgecolors='k', s=80)
    axes[1].set_title("2. Valor Predicho por el Perceptrón")
    axes[1].set_xlabel("Entrada X1")
    axes[1].set_ylabel("Entrada X2" if len(entradas[0]) > 1 else "Cero")
    fig.colorbar(sc2, ax=axes[1])

    # Gráfico 3: Coincidencia
    axes[2].scatter(x1, x2, c=colores, edgecolors='k', s=80)
    axes[2].set_title("3. Coincidencia (Verde=Acierto, Rojo=Fallo)")
    axes[2].set_xlabel("Entrada X1")
    axes[2].set_ylabel("Entrada X2" if len(entradas[0]) > 1 else "Cero")

    plt.tight_layout()
    plt.show()


def main():
    print("=== PROGRAMA DE INTERACCIÓN CON PERCEPTRÓN MANUAL ===")
    ruta_csv = input("Ingrese la ruta del archivo CSV con los datos: ").strip()

    try:
        entradas, esperados = cargar_csv(ruta_csv)
    except Exception as e:
        print(f"Error al intentar abrir el archivo: {e}")
        return

    total_vectores = len(entradas)
    if total_vectores == 0:
        print("El archivo CSV especificado no contiene datos válidos.")
        return

    num_entradas = len(entradas[0])
    print(f"\n[OK] Datos cargados correctamente: {total_vectores} vectores de {num_entradas} entrada(s) cada uno.")

    while True:
        print("\n--- Ingreso de Pesos ---")
        try:
            w0 = float(input("Ingrese el peso para el sesgo (bias w0): "))
            pesos = []
            for i in range(num_entradas):
                w_i = float(input(f"Ingrese el peso w{i+1} para la entrada X{i+1}: "))
                pesos.append(w_i)
        except ValueError:
            print("Entrada inválida. Debe ingresar números reales.")
            continue

        print("\n--- Selección de Función de Activación ---")
        print("1. Escalón Unipolar (Salidas en {0, 1})")
        print("2. Escalón Bipolar / Signo (Salidas en {-1, 1})")
        opcion_func = input("Seleccione la función (1 o 2): ").strip()

        if opcion_func == '2':
            func_act = activacion_escalon_bipolar
        else:
            func_act = activacion_escalon_unipolar

        # Evaluación de los vectores de entrada
        predicciones, colores, aciertos = evaluar_datos(
            entradas, esperados, pesos, w0, func_act
        )

        porcentaje = (aciertos / total_vectores) * 100
        print(f"\nResultados: {aciertos}/{total_vectores} aciertos ({porcentaje:.2f}% de precisión).")

        # Generación de gráficos
        graficar_resultados(entradas, esperados, predicciones, colores)

        # Preguntar al usuario si desea volver a probar
        respuesta = input("\n¿Desea probar con otros pesos o función de activación? (s/n): ").strip().lower()
        if respuesta != 's':
            print("Programa finalizado.")
            break


if __name__ == '__main__':
    main()