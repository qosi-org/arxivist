from setuptools import find_packages, setup

setup(
    name="asset_pricing_ml",
    version="1.0.0",
    description="Empirical Asset Pricing via Machine Learning — Gu, Kelly, Xiu (RFS 2020)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0", "numpy>=1.24.0", "pandas>=2.0.0",
        "scikit-learn>=1.3.0", "scipy>=1.11.0", "pyyaml>=6.0",
        "matplotlib>=3.7.0", "statsmodels>=0.14.0",
    ],
)
