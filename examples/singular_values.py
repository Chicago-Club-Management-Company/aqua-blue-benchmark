import numpy as np

import aqua_blue_benchmark


def main():

    rng = np.random.default_rng(seed=0)
    matrix = rng.uniform(size=(100, 50))
    bar = aqua_blue_benchmark.bar.Bar(matrix=matrix)
    singular_values = bar.singular_values(phrase="AAAAAAAAAA")
    print(singular_values)


if __name__ == "__main__":

    main()
