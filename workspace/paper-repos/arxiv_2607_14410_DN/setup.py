from setuptools import find_packages, setup

setup(
    name="lattice",
    version="0.1.0",
    description="ArXivist reproduction of LATTICE (arXiv:2607.14410)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
