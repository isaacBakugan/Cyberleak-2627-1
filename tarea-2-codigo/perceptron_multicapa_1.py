# Nombre del integrante:Andres Arrieta
# Cédula del integrante:30496624

import numpy as np
import matplotlib.pyplot as plt

sig = lambda z: 1 / (1 + np.exp(-z))

def leer(ruta):
    d = np.genfromtxt(ruta, delimiter=",")
    d = d[~np.isnan(d).any(axis=1)]
    return d[:, :-1], d[:, -1]

class Red:
    def __init__(self, u, v, L, b, e=0, W=None, B=None, cl=None):
        self.u, self.v, self.L, self.b, self.e, self.cl = u, v, L, b, e, cl
        t = [u] + [b] * L + [v]
        self.W = W or [np.random.randn(t[i + 1], t[i]) / np.sqrt(t[i]) for i in range(L + 1)]
        self.B = B or [np.zeros(t[i + 1]) for i in range(L + 1)]

    def adelante(self, X):
        A = [X]
        for W, B in zip(self.W, self.B):
            A.append(sig(A[-1] @ W.T + B))
        return A

    def entrenar(self, X, T, epocas, lr=0.5):
        err = []
        for _ in range(epocas):
            for i in np.random.permutation(len(X)):
                A = self.adelante(X[i:i + 1])
                d = (A[-1] - T[i:i + 1]) * A[-1] * (1 - A[-1])
                for k in range(self.L, -1, -1):
                    gW, gB, W = d.T @ A[k], d[0], self.W[k]
                    d = (d @ W) * A[k] * (1 - A[k]) if k else d
                    self.W[k], self.B[k] = W - lr * gW, self.B[k] - lr * gB
            err.append(np.mean((self.adelante(X)[-1] - T) ** 2))
            self.e += 1
        return err

    def guardar(self, ruta):
        c = max(self.u, self.b)
        f = [["tipo", "capa", "posicion"] + [f"w{i}" for i in range(c)] + ["sesgo"]]
        f += [[s, h] + [""] * (c + 1) + [";".join(f"{x:g}" for x in self.cl) if s == "v" else ""]
              for s, h in zip("uvLbe", [self.u, self.v, self.L, self.b, self.e])]
        f += [["o" if k == self.L else "h", k + 1, j] + list(W[j]) + [""] * (c - W.shape[1]) + [B[j]]
              for k, (W, B) in enumerate(zip(self.W, self.B)) for j in range(len(W))]
        np.savetxt(ruta, np.array(f, dtype=str), fmt="%s", delimiter=",")

    @staticmethod
    def cargar(ruta):
        filas = [l.rstrip("\r\n").split(",") for l in open(ruta, encoding="utf-8-sig") if l.strip()]
        hp = {f[0]: int(float(f[1])) for f in filas if f[0] in tuple("uvLbe")}
        cl = next((np.sort([float(x) for x in f[-1].split(";")]) for f in filas
                   if f[0] == "v" and len(f) > 2 and f[-1].strip()), None)
        N = {(f[0], int(f[1]), int(f[2])): f[3:] for f in filas if f[0] in ("h", "o")}
        u, v, L, b = hp["u"], hp["v"], hp["L"], hp["b"]
        c, t = max(u, b), [u] + [b] * L + [v]
        fila = lambda k, j: N["ho"[k == L], k + 1, j]
        sesgo = lambda r: float(r[c]) if len(r) > c and r[c].strip() else 0.0
        W = [np.array([[float(x) for x in fila(k, j)[:t[k]]] for j in range(t[k + 1])]) for k in range(L + 1)]
        B = [np.array([sesgo(fila(k, j)) for j in range(t[k + 1])]) for k in range(L + 1)]
        return Red(u, v, L, b, hp["e"], W, B, cl)

def datos(red, ruta):
    X, y = leer(ruta)
    cl = red.cl if red.cl is not None else np.unique(y)
    if X.shape[1] != red.u or len(cl) != red.v or not np.isin(y, cl).all():
        raise ValueError(f"el archivo tiene {X.shape[1]} columnas de entrada y {len(np.unique(y))} clases, pero la red "
                         f"tiene {red.u} neuronas de entrada y {red.v} de salida.\nLa red debe configurarse con "
                         f"{X.shape[1]} neuronas de entrada y {len(np.unique(y))} neuronas de salida.")
    red.cl = cl
    return X, np.searchsorted(cl, y)

def graficar(X, real, pred, cl, err, e):
    fig = plt.figure(figsize=(17, 5))
    d3 = X.shape[1] >= 3
    ejes = [X[:, i] if i < X.shape[1] else np.zeros(len(X)) for i in range(3 if d3 else 2)]
    for pos, col, tit in [(1, real, "Datos (clase esperada)"),
                          (2, np.where(real == pred, "green", "red"), "Prediccion (verde=correcta, rojo=incorrecta)")]:
        ax = fig.add_subplot(1, 3, pos, projection="3d" if d3 else None)
        sc = ax.scatter(*ejes, c=col)
        ax.set_title(tit)
        if pos == 1:
            ax.legend(sc.legend_elements()[0], [f"{c:g}" for c in cl], title="Clase")
    ax = fig.add_subplot(1, 3, 3)
    if err:
        ax.plot(range(e - len(err) + 1, e + 1), err)
        ax.set(title="Error por epoca", xlabel="Epoca", ylabel="Error cuadratico medio")
    else:
        M = np.zeros((len(cl), len(cl)), int)
        np.add.at(M, (real, pred), 1)
        ax.matshow(M)
        for (i, j), n in np.ndenumerate(M):
            ax.text(j, i, n, ha="center", va="center", color="red")
        et = [f"{c:g}" for c in cl]
        ax.set(title="Matriz de confusion", xlabel="Clase predicha", ylabel="Clase real",
               xticks=range(len(cl)), yticks=range(len(cl)), xticklabels=et, yticklabels=et)
    plt.tight_layout()
    plt.show()

def pedir(msg):
    v = input(msg)
    return int(v) if v.isdigit() and int(v) > 0 else pedir(msg)

def main():
    red = None
    while red is None:
        op = input("1) Crear un nuevo perceptron multicapa\n2) Cargar un perceptron desde un archivo\nOpcion: ").strip()
        try:
            if op == "1":
                red = Red(pedir("Neuronas de entrada: "), pedir("Neuronas de salida: "),
                          pedir("Capas ocultas: "), pedir("Neuronas por capa: "))
            elif op == "2":
                red = Red.cargar(input("Archivo del perceptron: ").strip())
        except Exception as ex:
            print(f"No se pudo cargar la red: {ex}")
    hiper = lambda: print(f"Hiperparametros -> u (entradas): {red.u} | v (salidas): {red.v} | "
                          f"L (capas ocultas): {red.L} | b (neuronas por capa): {red.b} | e (epocas): {red.e}")
    hiper()
    while (op := input("\n1) Entrenar\n2) Probar\n3) Guardar\n4) Salir\nOpcion: ").strip()) != "4":
        try:
            if op in ("1", "2"):
                X, idx = datos(red, input("Archivo de datos de " + ("entrenamiento: " if op == "1" else "prueba: ")).strip())
                err = red.entrenar(X, np.eye(red.v)[idx], pedir("Numero de epocas: ")) if op == "1" else None
                pred = np.argmax(red.adelante(X)[-1], axis=1)
                print(f"Aciertos: {np.sum(pred == idx)}/{len(X)} ({100 * np.mean(pred == idx):.1f}%)")
                hiper()
                graficar(X, idx, pred, red.cl, err, red.e)
            elif op == "3":
                ruta = input("Nombre del archivo donde guardar la red: ").strip()
                ruta += "" if ruta.endswith(".csv") else ".csv"
                red.guardar(ruta)
                print(f"Red guardada en {ruta}")
            else:
                print("Opcion invalida.")
        except (OSError, ValueError) as ex:
            print(f"Error: {ex}")

if __name__ == "__main__":
    main()
