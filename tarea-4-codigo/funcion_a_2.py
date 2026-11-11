# Function a: y = sin^2(x) / x
import math

BOUNDS = (-10, 10)        # limits of every variable
VARIABLE_COUNT = 1        # 1 for function a, 2 for function b


def fitness(*variables):  # higher is better
    x = variables[0]
    if x == 0:
        return 0.0  # 0/0 is undefined; the limit when x -> 0 is 0
    return math.sin(x) ** 2 / x
