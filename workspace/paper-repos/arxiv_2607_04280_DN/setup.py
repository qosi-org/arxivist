from setuptools import find_packages, setup

setup(
    name="sqrt_law_abm",
    version="0.1.0",
    description=(
        "ArXivist reproduction of 'Order Splitting and Liquidity Replenishment Are "
        "Jointly Necessary for the Square-Root Law of Market Impact' (arXiv:2607.04280)"
    ),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26,<2",
        "torch>=2.2,<3",
        "pandas>=2.2,<3",
        "scipy>=1.12,<2",
        "matplotlib>=3.8,<4",
        "pyyaml>=6.0,<7",
        "joblib>=1.3,<2",
        "tqdm>=4.66,<5",
    ],
)
