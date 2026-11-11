# Nombre del integrante: Richard Yrady
# Cédula del integrante: 30715365

"""
Genetic algorithm that looks for the maximum of a function.

"""

import importlib.util
import json
import math
import os
import random
import statistics

ELITE_COUNT = 2
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.9
BLX_ALPHA = 0.5
MUTATION_PROBABILITY = 0.5

DEFAULT_PARAMETERS = {
    "population_size": 10,
    "difference_threshold": 0.05,
    "max_generations": 500,
    "mutation_variability": 1.0,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- function file

def find_file(path, extra_dirs=()):
    """Return an existing path for 'path', also looking next to this script."""
    if os.path.isfile(path):
        return path
    for folder in list(extra_dirs) + [SCRIPT_DIR]:
        candidate = os.path.join(folder, path)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(path)


def load_function(path, extra_dirs=()):
    """Import a function file and check that it follows the contract."""
    full_path = find_file(path, extra_dirs)
    name = os.path.splitext(os.path.basename(full_path))[0]
    spec = importlib.util.spec_from_file_location(name, full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attribute in ("BOUNDS", "VARIABLE_COUNT", "fitness"):
        if not hasattr(module, attribute):
            raise ValueError(f"{os.path.basename(full_path)} has no {attribute}")
    module.file_name = os.path.basename(full_path)
    return module


def evaluate(function, genes):
    """Fitness of one member. Points where the function fails rank last."""
    try:
        value = float(function.fitness(*genes))
    except (ZeroDivisionError, ValueError, OverflowError):
        return -math.inf
    return value if math.isfinite(value) else -math.inf


# ------------------------------------------------------------------ population

class Population:
    def __init__(self, function, parameters, members=None, generations_elapsed=0):
        self.function = function
        self.parameters = dict(parameters)
        self.generations_elapsed = generations_elapsed
        low, high = function.BOUNDS
        if members is None:
            members = [[random.uniform(low, high) for _ in range(function.VARIABLE_COUNT)]
                       for _ in range(parameters["population_size"])]
        self.members = [(genes, evaluate(function, genes)) for genes in members]
        self.sort()

    def sort(self):
        self.members.sort(key=lambda member: member[1], reverse=True)

    def fitness_values(self):
        return [fit for _, fit in self.members]

    def difference(self):
        values = self.fitness_values()
        return values[0] - values[-1]

    def converged(self):
        return self.difference() < self.parameters["difference_threshold"]

    # --- genetic operators

    def tournament(self):
        contestants = random.sample(self.members, min(TOURNAMENT_SIZE, len(self.members)))
        return max(contestants, key=lambda member: member[1])[0]

    def crossover(self, parent_1, parent_2):
        if random.random() > CROSSOVER_PROBABILITY:
            return list(parent_1)
        child = []
        for gene_1, gene_2 in zip(parent_1, parent_2):
            low, high = min(gene_1, gene_2), max(gene_1, gene_2)
            spread = (high - low) * BLX_ALPHA
            child.append(random.uniform(low - spread, high + spread))
        return child

    def mutate(self, genes):
        low, high = self.function.BOUNDS
        sigma = self.parameters["mutation_variability"]
        mutated = []
        for gene in genes:
            if random.random() < MUTATION_PROBABILITY:
                gene += random.gauss(0, sigma)
            mutated.append(min(max(gene, low), high))
        return mutated

    def next_generation(self):
        size = self.parameters["population_size"]
        new_members = [list(genes) for genes, _ in self.members[:min(ELITE_COUNT, size)]]
        while len(new_members) < size:
            child = self.crossover(self.tournament(), self.tournament())
            new_members.append(self.mutate(child))
        self.members = [(genes, evaluate(self.function, genes)) for genes in new_members]
        self.sort()
        self.generations_elapsed += 1

    # --- training and output

    def train(self):
        print_header()
        while self.generations_elapsed < self.parameters["max_generations"]:
            self.next_generation()
            values = self.fitness_values()
            print(f"{self.generations_elapsed:>10} | {values[0]:>14.6f} | "
                  f"{statistics.median(values):>14.6f} | {values[-1]:>14.6f}")
            if self.converged():
                print(f"\nLa población convergió: max - min = {self.difference():.6f} "
                      f"< umbral {self.parameters['difference_threshold']}.")
                return
        print("\nSe alcanzó el número máximo de generaciones.")

    def show(self):
        p = self.parameters
        print("\n=== Parámetros ===")
        print(f"Función:                    {self.function.file_name}")
        print(f"Tamaño de la población:     {p['population_size']}")
        print(f"Umbral de diferencia:       {p['difference_threshold']}")
        print(f"Generaciones transcurridas: {self.generations_elapsed} "
              f"(máximo {p['max_generations']})")
        print(f"Variabilidad de mutación:   {p['mutation_variability']}")
        print("\n=== Miembros (de mejor a peor) ===")
        names = variable_names(self.function.VARIABLE_COUNT)
        for rank, (genes, fit) in enumerate(self.members, start=1):
            variables = ", ".join(f"{n} = {g:.6f}" for n, g in zip(names, genes))
            print(f"{rank:>3}. {variables}  ->  f = {fit:.6f}")

    def save(self, path):
        data = {
            "function": self.function.file_name,
            "parameters": {**self.parameters, "generations_elapsed": self.generations_elapsed},
            "members": [
                {"variables": list(genes),
                 "fitness": fit if math.isfinite(fit) else None,
                 "genes": {"representation": "real", "values": list(genes)}}
                for genes, fit in self.members
            ],
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)


def load_population(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    function = load_function(data["function"], [os.path.dirname(os.path.abspath(path))])
    parameters = {key: data["parameters"][key] for key in DEFAULT_PARAMETERS}
    members = []
    for member in data["members"]:
        genes = member.get("genes")
        if isinstance(genes, dict) and genes.get("representation") == "real":
            members.append([float(v) for v in genes["values"]])
        else:
            members.append([float(v) for v in member["variables"]])
    if len(members) != parameters["population_size"]:
        raise ValueError("members does not match population_size")
    if any(len(genes) != function.VARIABLE_COUNT for genes in members):
        raise ValueError("members do not match the function's VARIABLE_COUNT")
    return Population(function, parameters, members,
                      data["parameters"].get("generations_elapsed", 0))


def variable_names(count):
    return ["x", "y"] if count == 2 else ["x"] if count == 1 else [f"x{i + 1}" for i in range(count)]


def print_header():
    print(f"\n{'Generación':>10} | {'Máximo':>14} | {'Mediana':>14} | {'Mínimo':>14}")
    print("-" * 62)


# ------------------------------------------------------------------------ menu

def ask_number(message, default, cast, minimum=None, exclusive_minimum=None):
    while True:
        text = input(f"{message} [{default}]: ").strip()
        if text == "":
            return default
        try:
            value = cast(text)
        except ValueError:
            print("  Valor inválido.")
            continue
        if minimum is not None and value < minimum:
            print(f"  Debe ser mayor o igual a {minimum}.")
        elif exclusive_minimum is not None and value <= exclusive_minimum:
            print(f"  Debe ser mayor que {exclusive_minimum}.")
        else:
            return value


def create_population():
    function = load_function(input("Archivo de la función (ej. funcion_a_2.py): ").strip())
    parameters = {
        "population_size": ask_number("Tamaño de la población",
                                      DEFAULT_PARAMETERS["population_size"], int, minimum=2),
        "difference_threshold": ask_number("Umbral de diferencia",
                                           DEFAULT_PARAMETERS["difference_threshold"], float,
                                           minimum=0),
        "max_generations": ask_number("Número máximo de generaciones",
                                      DEFAULT_PARAMETERS["max_generations"], int, minimum=1),
        "mutation_variability": ask_number("Variabilidad de la mutación",
                                           DEFAULT_PARAMETERS["mutation_variability"], float,
                                           exclusive_minimum=0),
    }
    population = Population(function, parameters)
    print(f"\nPoblación creada con {parameters['population_size']} miembros aleatorios "
          f"en {function.BOUNDS}.")
    return population


def train_population(population):
    if population.generations_elapsed >= population.parameters["max_generations"]:
        extra = ask_number("Ya se alcanzó el máximo de generaciones. ¿Cuántas más?", 100, int,
                           minimum=1)
        population.parameters["max_generations"] += extra
    population.train()
    population.show()


def main():
    population = None
    while True:
        print("\n=== Algoritmo genético ===")
        print("1. Crear población")
        print("2. Cargar población de archivo")
        if population is not None:
            print("3. Entrenar")
            print("4. Mostrar población")
            print("5. Guardar población")
        print("0. Salir")
        try:
            option = input("Opción: ").strip()
        except EOFError:
            break
        try:
            if option == "1":
                population = create_population()
            elif option == "2":
                population = load_population(input("Archivo de la población: ").strip())
                population.show()
            elif option == "3" and population is not None:
                train_population(population)
            elif option == "4" and population is not None:
                population.show()
            elif option == "5" and population is not None:
                path = input("Archivo donde guardar (ej. population_2_a_low.json): ").strip()
                population.save(path)
                print(f"Población guardada en {path}.")
            elif option == "0":
                break
            else:
                print("Opción inválida.")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
