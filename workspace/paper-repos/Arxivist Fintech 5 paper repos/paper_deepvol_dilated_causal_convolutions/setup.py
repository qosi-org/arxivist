"""Setup file for the deepvol package."""
from setuptools import setup, find_packages

setup(
    name="deepvol",
    version="0.1.0",
    description="DeepVol: Volatility forecasting from high-frequency data with dilated causal convolutions",
    author="Fernando Moreno-Pino, Stefan Zohren",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "arch>=6.2.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.65.0",
    ],
)
