"""
setup.py
=========
Editable-install setup for the arxivist_artemis package.
Reproduction repo for arXiv:2603.18107 (Ray, 2026).
"""

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="arxivist_artemis",
    version="0.1.0",
    description=(
        "Reproduction of 'ARTEMIS: A Neuro-Symbolic Framework for Economically "
        "Constrained Market Dynamics' (arXiv:2603.18107)"
    ),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=install_requires,
)
