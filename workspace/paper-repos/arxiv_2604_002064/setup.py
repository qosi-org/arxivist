"""Setup script for the noisy_qnn_uat package.

ArXivist-generated repository for:
"Quantitative Universal Approximation for Noisy Quantum Neural Networks"
(Gonon, Jacquier, Mordarski, arXiv:2604.02064v3)
"""

from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="noisy_qnn_uat",
    version="0.1.0",
    description=(
        "Reproduction of 'Quantitative Universal Approximation for Noisy Quantum "
        "Neural Networks' (arXiv:2604.02064) -- QNN circuit, depolarising-noise "
        "bounds, and Black-Scholes Put pricing experiments."
    ),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=install_requires,
)
