[![Custom shields.io](https://img.shields.io/badge/docs-brightgreen?logo=github&logoColor=green&label=gh-pages)](https://chicago-club-management-company.github.io/aqua-blue-benchmark/)

 You can run and install with:

```bash
git clone https://github.com/Chicago-Club-Management-Company/aqua-blue-benchmark
pip install aqua-blue-benchmark/
```

or, with dev packages, and running the CI/CD:

```bash
git clone https://github.com/Chicago-Club-Management-Company/aqua-blue-benchmark
cd aqua-blue-benchmark/
pip install aqua-blue-benchmark/[dev]
pytest
ruff check aqua_blue_benchmark/
mypy aqua_blue_benchmark/
```