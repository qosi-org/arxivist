from setuptools import find_packages, setup

setup(
    name="kyle-liquidity",
    version="0.1.0",
    description="Numerical verification suite for arXiv:2607.10934 (Ekren, Nikitopoulos, Vy)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26,<2.1",
        "scipy>=1.13,<1.14",
        "pyyaml>=6.0,<7.0",
    ],
)
