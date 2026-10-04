# Nombre del integrante: Andres Arrieta
# Cédula del integrante: 30496624

# haga su tarea aqui
import matplotlib.pyplot as plt

def cargar_csv(ruta):
    filas = [l.strip().split(",") for l in open(ruta, encoding="utf-8-sig") if l.strip()]
    try:
        [float(v) for v in filas[0]]
    except ValueError:
        filas = filas[1:]
    datos = [[float(v) for v in f] for f in filas]
    return [f[:-1] for f in datos], [f[-1] for f in datos]

def suma(pesos, entrada, sesgo):
    return sesgo + sum(w * x for w, x in zip(pesos, entrada))

def exp(x, k=20):
    r, t = 1.0, 1.0
    for i in range(1, 30):
        t *= (x / k) / i
        r += t
    return r ** k

escalon = lambda n: 1.0 if n >= 0 else 0.0
sigmoide = lambda n: 1.0 / (1.0 + exp(-n))
clase = lambda v: 1 if v >= 0.5 else 0

def graficar(X, esp, pred, n):
    x1 = [f[0] for f in X]
    x2 = [f[1] if n >= 2 else 0.0 for f in X]
    ce = [clase(v) for v in esp]
    cp = [clase(v) for v in pred]
    col = lambda c: ["tab:blue" if v == 1 else "tab:orange" for v in c]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    titulos = ["Valor esperado", "Valor predicho por el perceptron", "Coincidencia (verde=si, rojo=no)"]
    colores = [col(ce), col(cp), ["green" if a == b else "red" for a, b in zip(ce, cp)]]
    for a, t, c in zip(ax, titulos, colores):
        a.scatter(x1, x2, c=c)
        a.set_title(t)
        a.set_xlabel("x1")
        a.set_ylabel("x2")
    plt.tight_layout()
    plt.show()

def main():
    X, Y = cargar_csv(input("Ingrese la ruta del archivo CSV: ").strip())
    n = len(X[0])
    print(f"Se cargaron {len(X)} vectores con {n} entrada(s) cada uno.")
    seguir = "s"
    while seguir == "s":
        sesgo = float(input("Peso del sesgo (bias): "))
        pesos = [float(input(f"Peso para la columna x{i+1}: ")) for i in range(n)]
        act = sigmoide if input("Activacion (1=Escalon, 2=Sigmoide): ").strip() == "2" else escalon
        pred = [act(suma(pesos, x, sesgo)) for x in X]
        aciertos = sum(clase(p) == clase(e) for p, e in zip(pred, Y))
        print(f"Aciertos: {aciertos}/{len(X)} ({100*aciertos/len(X):.1f}%)")
        graficar(X, Y, pred, n)
        seguir = input("¿Desea probar con otros pesos? (s/n): ").strip().lower()

if __name__ == "__main__":
    main()