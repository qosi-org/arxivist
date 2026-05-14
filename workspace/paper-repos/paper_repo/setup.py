from setuptools import setup, find_packages

setup(
    name="volsig",
    version="0.1.0",
    description="Signature-based implied volatility calibration (Alòs et al. 2026)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "torch>=2.1.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
