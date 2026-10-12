# Nombre del integrante: Richard Yrady
# Cédula del integrante: 30715365

# haga su tarea aqui
import csv
import numpy as np
import matplotlib.pyplot as plt
 
TASA = 0.3  
 
 
def sigmoide(z):
    return 1 / (1 + np.exp(-z))
 
 
 
class Red:
    def __init__(self, u, v, L, b, e=0):
        self.u = u
        self.v = v
        self.L = L
        self.b = b
        self.e = e
 
        tamanos = [u] + [b] * L + [v]
 
        self.pesos = []
        self.sesgos = []
        for k in range(len(tamanos) - 1):
            self.pesos.append(np.random.randn(tamanos[k + 1], tamanos[k]) * 0.5)
            self.sesgos.append(np.random.randn(tamanos[k + 1]) * 0.5)
 
    def adelante(self, x):
        salidas = [x]
        for k in range(len(self.pesos)):
            z = self.pesos[k] @ salidas[-1] + self.sesgos[k]
            salidas.append(sigmoide(z))
        return salidas
 
    def predecir(self, X):
        predichas = []
        for x in X:
            salida = self.adelante(x)[-1]
            predichas.append(np.argmax(salida))
        return np.array(predichas)
 
    def entrenar(self, X, clases_idx, epocas):
        errores = []
        for epoca in range(epocas):
            error_total = 0
            for i in np.random.permutation(len(X)):
                deseada = np.zeros(self.v)
                deseada[clases_idx[i]] = 1
 
                salidas = self.adelante(X[i])
                error_total += np.sum((salidas[-1] - deseada) ** 2)
 
                delta = (salidas[-1] - deseada) * salidas[-1] * (1 - salidas[-1])

                for k in range(len(self.pesos) - 1, -1, -1):
                    delta_anterior = (self.pesos[k].T @ delta) * salidas[k] * (1 - salidas[k])
                    self.pesos[k] -= TASA * np.outer(delta, salidas[k])
                    self.sesgos[k] -= TASA * delta
                    delta = delta_anterior
 
            errores.append(error_total / len(X))
        self.e += epocas
        return errores
 
    def mostrar_hiperparametros(self):
        print("\nHiperparametros:")
        print("  Neuronas de entrada:", self.u)
        print("  Neuronas de salida:", self.v)
        print("  Capas ocultas:", self.L)
        print("  Neuronas por capa oculta:", self.b)
        print("  Epocas entrenadas:", self.e)
 
    def guardar(self, ruta):
        ancho = max(self.u, self.b) + 1 
        with open(ruta, "w", newline="") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(["tipo", "capa", "posicion"] + ["peso"] * ancho)
 
            escritor.writerow(["u", self.u] + [""] * (ancho + 1))
            escritor.writerow(["v", self.v] + [""] * (ancho + 1))
            escritor.writerow(["L", self.L] + [""] * (ancho + 1))
            escritor.writerow(["b", self.b] + [""] * (ancho + 1))
            escritor.writerow(["e", self.e] + [""] * (ancho + 1))
 
            for k in range(len(self.pesos)):
                if k == len(self.pesos) - 1:
                    tipo = "o"
                else:
                    tipo = "h"
                for pos in range(len(self.pesos[k])):
                    fila = [tipo, k + 1, pos] + list(self.pesos[k][pos]) + [self.sesgos[k][pos]]
                    fila = fila + [""] * (ancho + 3 - len(fila))
                    escritor.writerow(fila)
 
 
def cargar_red(ruta):
    with open(ruta, newline="") as archivo:
        filas = list(csv.reader(archivo))
 
    hiper = {}
    for fila in filas[1:6]:
        hiper[fila[0]] = int(fila[1])
    red = Red(hiper["u"], hiper["v"], hiper["L"], hiper["b"], hiper["e"])
 
    tamanos = [red.u] + [red.b] * red.L + [red.v]
    for fila in filas[6:]:
        capa = int(fila[1]) - 1
        pos = int(fila[2])
        valores = [float(x) for x in fila[3:3 + tamanos[capa] + 1]]
        red.pesos[capa][pos] = valores[:-1]
        red.sesgos[capa][pos] = valores[-1]
    return red
 
 
 
def cargar_datos(ruta, red):
    datos = np.loadtxt(ruta, delimiter=",", skiprows=1)
    X = datos[:, :-1]
    y = datos[:, -1]
    clases = list(np.unique(y))
 
    if X.shape[1] != red.u or len(clases) != red.v:
        raise ValueError(
            "El archivo tiene " + str(X.shape[1]) + " entradas y " + str(len(clases))
            + " clases. Configure la red con " + str(X.shape[1])
            + " neuronas de entrada y " + str(len(clases)) + " de salida."
        )
 
    clases_idx = np.array([clases.index(valor) for valor in y])
    return X, clases_idx, clases
 
 
 
def graficar(X, clases_idx, clases, predichas, errores=None):
    if X.shape[1] >= 3:
        proyeccion = "3d"
    else:
        proyeccion = None

    colores = []
    for i in range(len(X)):
        if predichas[i] == clases_idx[i]:
            colores.append("green")
        else:
            colores.append("red")
 
    fig = plt.figure(figsize=(17, 5))
 
    ax1 = fig.add_subplot(1, 3, 1, projection=proyeccion)
    if proyeccion == "3d":
        ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=clases_idx)
    else:
        ax1.scatter(X[:, 0], X[:, 1], c=clases_idx)
    ax1.set_title("Datos del archivo")

    ax2 = fig.add_subplot(1, 3, 2, projection=proyeccion)
    if proyeccion == "3d":
        ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=colores)
    else:
        ax2.scatter(X[:, 0], X[:, 1], c=colores)
    ax2.set_title("Prediccion (verde = correcta, rojo = incorrecta)")

    ax3 = fig.add_subplot(1, 3, 3)
    if errores is not None:
        ax3.plot(errores)
        ax3.set_title("Error por epoca")
        ax3.set_xlabel("Epoca")
        ax3.set_ylabel("Error")
    else:
        n = len(clases)
        matriz = np.zeros((n, n), dtype=int)
        for i in range(len(X)):
            matriz[clases_idx[i], predichas[i]] += 1
        ax3.matshow(matriz)
        for fila in range(n):
            for col in range(n):
                ax3.text(col, fila, matriz[fila, col], ha="center", va="center", color="red")
        ax3.set_xticks(range(n))
        ax3.set_xticklabels([int(c) for c in clases])
        ax3.set_yticks(range(n))
        ax3.set_yticklabels([int(c) for c in clases])
        ax3.set_title("Matriz de confusion\n(filas = real, columnas = predicho)", y=1.12)
 
    plt.tight_layout()
    plt.show()
 

 
def pedir_entero(mensaje, minimo):
    while True:
        try:
            n = int(input(mensaje))
            if n >= minimo:
                return n
            print("El numero debe ser al menos", minimo)
        except ValueError:
            print("Por favor ingrese un numero entero.")
 
 
def crear_red():
    u = pedir_entero("Numero de neuronas de entrada: ", 2)
    v = pedir_entero("Numero de neuronas de salida: ", 1)
    L = pedir_entero("Numero de capas ocultas: ", 0)
    b = pedir_entero("Numero de neuronas por capa oculta: ", 1)
    return Red(u, v, L, b)
 
 
 
def main():
    print("=== Perceptron multicapa ===")
    red = None
    while red is None:
        print("1) Crear un nuevo perceptron multicapa")
        print("2) Cargar un perceptron multicapa de un archivo")
        opcion = input("Opcion: ")
        if opcion == "1":
            red = crear_red()
        elif opcion == "2":
            try:
                red = cargar_red(input("Archivo de la red: "))
            except Exception as error:
                print("No se pudo cargar la red:", error)
 
    while True:
        print("\n--- Menu ---")
        print("1) Entrenar la red")
        print("2) Probar la red")
        print("3) Guardar la red")
        print("4) Salir")
        opcion = input("Opcion: ")
 
        try:
            if opcion == "1":
                ruta = input("Archivo de datos de entrenamiento: ")
                X, clases_idx, clases = cargar_datos(ruta, red)
                epocas = pedir_entero("Numero de epocas: ", 1)
                errores = red.entrenar(X, clases_idx, epocas)
                predichas = red.predecir(X)
                print("Aciertos:", np.sum(predichas == clases_idx), "de", len(X))
                red.mostrar_hiperparametros()
                graficar(X, clases_idx, clases, predichas, errores)
 
            elif opcion == "2":
                ruta = input("Archivo de datos de prueba: ")
                X, clases_idx, clases = cargar_datos(ruta, red)
                predichas = red.predecir(X)
                print("Aciertos:", np.sum(predichas == clases_idx), "de", len(X))
                red.mostrar_hiperparametros()
                graficar(X, clases_idx, clases, predichas)
 
            elif opcion == "3":
                red.guardar(input("Archivo donde guardar la red: "))
                print("Red guardada.")
 
            elif opcion == "4":
                print("Fin del programa.")
                break
 
            else:
                print("Opcion invalida.")
 
        except Exception as error:
            print("Error:", error)
 
 
if __name__ == "__main__":
    main()