from setuptools import setup, find_packages

setup(
    name="spg_uvm",
    version="0.1.0",
    description="Stochastic Policy Gradient Methods for the Uncertain Volatility Model",
    author="ArXivist (code generation from arXiv:2605.06670)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)
