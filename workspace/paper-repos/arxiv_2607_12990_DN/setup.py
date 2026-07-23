from setuptools import find_packages, setup

setup(
    name="quantum_cva",
    version="0.1.0",
    description="Reproduction of arXiv:2607.12990 -- Noise-Aware Quantum CVA on Real Quantum Hardware",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
