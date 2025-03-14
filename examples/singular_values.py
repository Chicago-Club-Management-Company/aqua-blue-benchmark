import numpy as np

import cool_package


def main():

    rng = np.random.default_rng(seed=0)
    matrix = rng.uniform(size=(100, 50))
    bar = cool_package.bar.Bar(matrix=matrix)
    singular_values = bar.singular_values(phrase="AAAAAAAAAA")
    print(singular_values)


if __name__ == "__main__":

    main()
