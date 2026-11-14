"""Function a: y = sin^2(x) / x, with x in [-10, 10]."""
import math

BOUNDS = (-10, 10)        # limits of every variable
VARIABLE_COUNT = 1        # one variable: x


def fitness(*variables):  # higher is better
    (x,) = variables
    if x == 0:
        return 0.0        # 0/0 is undefined; the limit when x -> 0 is 0
    return math.sin(x) ** 2 / x
