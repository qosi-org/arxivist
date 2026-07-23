"""
setup.py
=========
Editable-install setup for the arxivist_bootstrap package.
Reproduction repo for arXiv:2606.11859 (Baldoni, Sparviero, Viola, 2026).
"""

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="arxivist_bootstrap",
    version="0.1.0",
    description=(
        "Reproduction of 'Scenario Generation for Time Series and Curves: "
        "A Comparison of Nonparametric and Semiparametric Bootstrap' (arXiv:2606.11859)"
    ),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=install_requires,
)
