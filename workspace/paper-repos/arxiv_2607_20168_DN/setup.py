from setuptools import find_packages, setup

setup(
    name="qkernel_finance",
    version="0.1.0",
    description="Reproduction of arXiv:2607.20168 -- Quantum Kernels and the Cross-Section of Stock Returns",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
