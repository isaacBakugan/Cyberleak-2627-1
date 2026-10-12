# Nombre del integrante: Luis Rivera
# Cédula del integrante: 27773794

# haga su tarea aqui

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt


class PerceptronMulticapa:
    """Implementación de un Perceptrón Multicapa (MLP) con Retropropagación."""

    def __init__(self, u, v, L, b, e=0):
        self.u = int(u)  # Entradas
        self.v = int(v)  # Salidas
        self.L = int(L)  # Capas ocultas
        self.b = int(b)  # Neuronas por capa oculta
        self.e = int(e)  # Épocas acumuladas

        # Arquitectura de capas: [u, b, b, ..., b, v]
        self.layer_sizes = [self.u] + [self.b] * self.L + [self.v]

        # Inicialización de pesos y sesgos (Xavier/Glorot)
        self.W = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (n_in + n_out))
            w = np.random.uniform(-limit, limit, (n_out, n_in))
            bias = np.zeros((n_out, 1))
            self.W.append(w)
            self.biases.append(bias)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))

    @staticmethod
    def sigmoid_deriv(a):
        return a * (1.0 - a)

    def forward(self, X):
        """Alimentación hacia delante (Forward pass).
        X debe ser de dimensión (u, N).
        Retorna la lista de activaciones para cada capa.
        """
        activations = [X]
        curr = X
        for w, bias in zip(self.W, self.biases):
            z = np.dot(w, curr) + bias
            curr = self.sigmoid(z)
            activations.append(curr)
        return activations

    def train(self, X, Y_onehot, epochs, lr=0.1):
        """Entrenamiento con algoritmo de retropropagación (Backpropagation)."""
        N = X.shape[1]
        epoch_errors = []

        for _ in range(epochs):
            # Forward pass
            activations = self.forward(X)
            y_pred = activations[-1]

            # Error Cuadrático Medio (MSE) por época
            loss = 0.5 * np.mean(np.sum((y_pred - Y_onehot) ** 2, axis=0))
            epoch_errors.append(loss)

            # Retropropagación (Backpropagation)
            deltas = [None] * len(self.W)
            # Capa de salida
            deltas[-1] = (y_pred - Y_onehot) * self.sigmoid_deriv(y_pred)

            # Capas ocultas de atrás hacia adelante
            for l in range(len(self.W) - 2, -1, -1):
                deltas[l] = np.dot(self.W[l + 1].T, deltas[l + 1]) * self.sigmoid_deriv(
                    activations[l + 1]
                )

            # Actualización de pesos y sesgos
            for l in range(len(self.W)):
                dW = np.dot(deltas[l], activations[l].T) / N
                db = np.sum(deltas[l], axis=1, keepdims=True) / N
                self.W[l] -= lr * dW
                self.biases[l] -= lr * db

        self.e += epochs
        return epoch_errors

    def predict_classes(self, X, unique_classes):
        """Predice la clase original asignada a cada vector de entrada."""
        activations = self.forward(X)
        y_pred = activations[-1]  # (v, N)
        class_indices = np.argmax(y_pred, axis=0)
        return np.array([unique_classes[idx] for idx in class_indices])


# ==============================================================================
# LECTURA Y GUARDADO DE ARCHIVOS CSV
# ==============================================================================

def cargar_dataset_csv(ruta_csv):
    """Carga un archivo de datos (Train/Test) y separa entradas y etiquetas esperadas."""
    with open(ruta_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for r in reader:
            if not r:
                continue
            try:
                val_row = [float(x.strip()) for x in r if x.strip() != ""]
                rows.append(val_row)
            except ValueError:
                continue

    data = np.array(rows)
    if data.size == 0:
        raise ValueError("El archivo de datos está vacío o no tiene formato numérico.")

    X = data[:, :-1].T  # Matriz de entradas: forma (u, N)
    y_raw = data[:, -1]  # Vector de clases reales: forma (N,)
    return X, y_raw


def guardar_mlp_csv(mlp, ruta_archivo):
    """Guarda la red neuronal en el formato CSV exigido."""
    u, v, L, b, e = mlp.u, mlp.v, mlp.L, mlp.b, mlp.e
    c = max(u, b)

    # Fila de encabezado
    header = ["tipo", "capa", "posicion"] + [f"w{i}" for i in range(c + 1)]
    rows = [header]

    # Primeras 5 filas: Hiperparámetros
    hyperparams = [("u", u), ("v", v), ("L", L), ("b", b), ("e", e)]
    for letter, val in hyperparams:
        row = [letter, val] + [""] * (c + 2)
        rows.append(row)

    # Filas de neuronas
    for layer_idx in range(len(mlp.W)):
        is_output = layer_idx == len(mlp.W) - 1
        tipo = "o" if is_output else "h"
        capa_num = layer_idx + 1  # 1-based layer index

        W_layer = mlp.W[layer_idx]
        bias_layer = mlp.biases[layer_idx]

        n_neurons = W_layer.shape[0]
        for pos in range(n_neurons):
            w_bias = bias_layer[pos, 0]
            weights = W_layer[pos, :].tolist()
            all_w = [w_bias] + weights
            padding = [""] * ((c + 1) - len(all_w))
            row = [tipo, capa_num, pos] + [str(w) for w in all_w] + padding
            rows.append(row)

    with open(ruta_archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\n[OK] Red guardada exitosamente en: {ruta_archivo}")


def cargar_mlp_csv(ruta_archivo):
    """Carga una red neuronal desde un archivo CSV con el formato especificado."""
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        params = {}
        neuron_rows = []

        for row in reader:
            if not row or not any(row):
                continue
            col1 = row[0].strip()
            if col1 in ["u", "v", "L", "b", "e"]:
                params[col1] = int(row[1].strip())
            elif col1 in ["h", "o"]:
                neuron_rows.append(row)

    mlp = PerceptronMulticapa(
        u=params["u"], v=params["v"], L=params["L"], b=params["b"], e=params["e"]
    )

    # Agrupar neuronas por capa
    layers_dict = {}
    for row in neuron_rows:
        capa_num = int(row[1].strip())
        pos = int(row[2].strip())
        raw_weights = [float(w) for w in row[3:] if w.strip() != ""]
        if capa_num not in layers_dict:
            layers_dict[capa_num] = {}
        layers_dict[capa_num][pos] = raw_weights

    for layer_idx in range(len(mlp.W)):
        capa_num = layer_idx + 1
        pos_dict = layers_dict[capa_num]
        for pos in range(len(pos_dict)):
            all_w = pos_dict[pos]
            mlp.biases[layer_idx][pos, 0] = all_w[0]
            mlp.W[layer_idx][pos, :] = all_w[1:]

    return mlp


# ==============================================================================
# VISUALIZACIÓN Y GRÁFICOS
# ==============================================================================

def visualizar_resultados(X, y_real, y_pred, es_entrenamiento, loss_history=None, unique_classes=None):
    """Muestra los 3 gráficos requeridos usando matplotlib."""
    u_dim = X.shape[0]
    coinciden = (y_real == y_pred)
    colores_acierto = ["green" if c else "red" for c in coinciden]

    fig = plt.figure(figsize=(18, 5))

    # Gráfico 1: Datos reales de entrada
    if u_dim >= 3:
        ax1 = fig.add_subplot(131, projection="3d")
        sc1 = ax1.scatter(X[0, :], X[1, :], X[2, :], c=y_real, cmap="tab10", s=50)
        ax1.set_title("1. Datos de Entrada (3D)")
        ax1.set_xlabel("X1")
        ax1.set_ylabel("X2")
        ax1.set_zlabel("X3")
    else:
        ax1 = fig.add_subplot(131)
        x_secunda = X[1, :] if u_dim > 1 else np.zeros(X.shape[1])
        sc1 = ax1.scatter(X[0, :], x_secunda, c=y_real, cmap="tab10", s=50, edgecolors="k")
        ax1.set_title("1. Datos de Entrada (2D)")
        ax1.set_xlabel("X1")
        ax1.set_ylabel("X2" if u_dim > 1 else "0")

    fig.colorbar(sc1, ax=ax1)

    # Gráfico 2: Clasificación (Acierto = Verde, Fallo = Rojo)
    if u_dim >= 3:
        ax2 = fig.add_subplot(132, projection="3d")
        ax2.scatter(X[0, :], X[1, :], X[2, :], c=colores_acierto, s=50)
        ax2.set_title("2. Predicción (Verde=Acierto, Rojo=Fallo)")
        ax2.set_xlabel("X1")
        ax2.set_ylabel("X2")
        ax2.set_zlabel("X3")
    else:
        ax2 = fig.add_subplot(132)
        x_secunda = X[1, :] if u_dim > 1 else np.zeros(X.shape[1])
        ax2.scatter(X[0, :], x_secunda, c=colores_acierto, s=50, edgecolors="k")
        ax2.set_title("2. Predicción (Verde=Acierto, Rojo=Fallo)")
        ax2.set_xlabel("X1")
        ax2.set_ylabel("X2" if u_dim > 1 else "0")

    # Gráfico 3: Error de la Red / Matriz de Confusión
    ax3 = fig.add_subplot(133)
    if es_entrenamiento and loss_history is not None:
        ax3.plot(loss_history, color="blue", linewidth=2)
        ax3.set_title("3. Error de Entrenam. por Época")
        ax3.set_xlabel("Época")
        ax3.set_ylabel("Pérdida (MSE)")
        ax3.grid(True)
    else:
        # Calcular Matriz de Confusión
        classes = sorted(list(unique_classes))
        n_c = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        cm = np.zeros((n_c, n_c), dtype=int)

        for r, p in zip(y_real, y_pred):
            cm[class_to_idx[r]][class_to_idx[p]] += 1

        cax = ax3.matshow(cm, cmap="Blues")
        fig.colorbar(cax, ax=ax3)
        ax3.set_title("3. Matriz de Confusión (Prueba)")
        ax3.set_xlabel("Predicho")
        ax3.set_ylabel("Esperado")
        ax3.set_xticks(range(n_c))
        ax3.set_yticks(range(n_c))
        ax3.set_xticklabels([str(c) for c in classes])
        ax3.set_yticklabels([str(c) for c in classes])

        for i in range(n_c):
            for j in range(n_c):
                ax3.text(j, i, str(cm[i, j]), va="center", ha="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.show()


# ==============================================================================
# LÓGICA DE INTERFACCIÓN Y MENÚ
# ==============================================================================

def imprimir_hiperparametros(mlp):
    print("\n" + "=" * 45)
    print("      HIPERPARÁMETROS DE LA RED NEURONAL")
    print("=" * 45)
    print(f"  - Entradas (u)         : {mlp.u}")
    print(f"  - Salidas (v)          : {mlp.v}")
    print(f"  - Capas ocultas (L)   : {mlp.L}")
    print(f"  - Neuronas / capa (b) : {mlp.b}")
    print(f"  - Épocas acumuladas (e): {mlp.e}")
    print("=" * 45 + "\n")


def obtener_one_hot_y_clases(y_raw, v_esperado):
    """Obtiene las clases únicas y valida la compatibilidad con 'v'."""
    unique_classes = sorted(np.unique(y_raw))
    num_clases = len(unique_classes)

    if num_clases > v_esperado:
        raise ValueError(
            f"El archivo contiene {num_clases} clases distintas {unique_classes}, "
            f"pero la red fue configurada para v = {v_esperado} neuronas de salida."
        )

    # Mapeo a One-Hot
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    N = len(y_raw)
    Y_onehot = np.zeros((v_esperado, N))

    for idx, c in enumerate(y_raw):
        col_idx = class_to_idx[c]
        Y_onehot[col_idx, idx] = 1.0

    return Y_onehot, unique_classes


def main():
    print("==================================================")
    print("   TAREA 2: PERCEPTRÓN MULTICAPA (MLP) DESDE CERO")
    print("==================================================")

    mlp = None

    while mlp is None:
        print("\nOpciones de inicio:")
        print("1. Crear un nuevo perceptrón multicapa")
        print("2. Cargar perceptrón multicapa desde archivo CSV")
        op = input("Seleccione una opción (1 o 2): ").strip()

        if op == "1":
            try:
                u = int(input("Número de neuronas de entrada (u): "))
                v = int(input("Número de neuronas de salida (v): "))
                L = int(input("Número de capas ocultas (L): "))
                b = int(input("Número de neuronas por capa oculta (b): "))
                mlp = PerceptronMulticapa(u=u, v=v, L=L, b=b)
                print("\n[OK] Red neuronal creada correctamente.")
            except ValueError:
                print("[Error] Ingrese valores enteros válidos.")

        elif op == "2":
            ruta = input("Ingrese la ruta del archivo CSV de la red: ").strip().strip('"').strip("'")
            try:
                mlp = cargar_mlp_csv(ruta)
                print("\n[OK] Red neuronal cargada exitosamente.")
            except Exception as ex:
                print(f"[Error] No se pudo cargar el archivo: {ex}")
        else:
            print("[Error] Opción no válida.")

    # Menú Principal
    while True:
        imprimir_hiperparametros(mlp)

        print("--- MENÚ PRINCIPAL ---")
        print("1. Entrenar la red con retropropagación (Backpropagation)")
        print("2. Probar la red con alimentación hacia delante (Forward Pass)")
        print("3. Guardar la red")
        print("4. Salir")
        opcion = input("Seleccione una opción (1-4): ").strip()

        if opcion == "1":
            ruta_datos = input("Archivo de datos de entrenamiento (CSV): ").strip().strip('"').strip("'")
            try:
                X, y_raw = cargar_dataset_csv(ruta_datos)
            except Exception as ex:
                print(f"[Error] No se pudo cargar el archivo de datos: {ex}")
                continue

            # Validación de dimensiones
            u_dataset = X.shape[0]
            if u_dataset != mlp.u:
                print("\n" + "!" * 60)
                print(f"[ERROR DE CONFIGURACIÓN] El dataset tiene u = {u_dataset} entradas,")
                print(f"pero la red neuronal está configurada para u = {mlp.u} entradas.")
                print(f"Por favor, configure una red con 'u = {u_dataset}'.")
                print("!" * 60)
                continue

            try:
                Y_onehot, unique_classes = obtener_one_hot_y_clases(y_raw, mlp.v)
            except ValueError as ex:
                print("\n" + "!" * 60)
                print(f"[ERROR DE CONFIGURACIÓN] {ex}")
                print(f"Por favor, reconfigure una red con 'v' acorde al archivo.")
                print("!" * 60)
                continue

            try:
                epocas = int(input("Número de épocas a entrenar: "))
            except ValueError:
                print("[Error] Debe ingresar un número entero para las épocas.")
                continue

            print("\nEntrenando la red...")
            loss_history = mlp.train(X, Y_onehot, epochs=epocas, lr=0.1)

            y_pred = mlp.predict_classes(X, unique_classes)
            aciertos = np.sum(y_raw == y_pred)
            precision = (aciertos / len(y_raw)) * 100
            print(f"\n[ENTRENAMIENTO COMPLETADO] Aciertos: {aciertos}/{len(y_raw)} ({precision:.2f}%)")

            visualizar_resultados(
                X, y_raw, y_pred, es_entrenamiento=True,
                loss_history=loss_history, unique_classes=unique_classes
            )

        elif opcion == "2":
            ruta_datos = input("Archivo de datos de prueba (CSV): ").strip().strip('"').strip("'")
            try:
                X, y_raw = cargar_dataset_csv(ruta_datos)
            except Exception as ex:
                print(f"[Error] No se pudo cargar el archivo de datos: {ex}")
                continue

            u_dataset = X.shape[0]
            if u_dataset != mlp.u:
                print("\n" + "!" * 60)
                print(f"[ERROR DE CONFIGURACIÓN] El dataset tiene u = {u_dataset} entradas,")
                print(f"pero la red neuronal está configurada para u = {mlp.u} entradas.")
                print("!" * 60)
                continue

            try:
                _, unique_classes = obtener_one_hot_y_clases(y_raw, mlp.v)
            except ValueError as ex:
                print("\n" + "!" * 60)
                print(f"[ERROR DE CONFIGURACIÓN] {ex}")
                print("!" * 60)
                continue

            y_pred = mlp.predict_classes(X, unique_classes)
            aciertos = np.sum(y_raw == y_pred)
            precision = (aciertos / len(y_raw)) * 100
            print(f"\n[PRUEBA COMPLETADA] Aciertos: {aciertos}/{len(y_raw)} ({precision:.2f}%)")

            visualizar_resultados(
                X, y_raw, y_pred, es_entrenamiento=False,
                unique_classes=unique_classes
            )

        elif opcion == "3":
            ruta_guardado = input("Ruta donde guardar la red (ej: red_entrenada.csv): ").strip().strip('"').strip("'")
            try:
                guardar_mlp_csv(mlp, ruta_guardado)
            except Exception as ex:
                print(f"[Error] No se pudo guardar la red: {ex}")

        elif opcion == "4":
            print("\nSaliendo del programa. ¡Éxito!")
            sys.exit(0)
        else:
            print("[Error] Opción no válida. Intente de nuevo.")


if __name__ == "__main__":
    main()