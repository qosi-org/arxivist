from setuptools import setup, find_packages

setup(
    name="dymgnn",
    version="0.1.0",
    description=(
        "Dynamic Multilayer GNN for Loan Default Prediction "
        "(Zandi et al., EJOR 2025)"
    ),
    author="ArXivist — generated from Zandi, Korangi, Óskarsdóttir, Mues, Bravo (2025)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0.1",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "xgboost": ["xgboost>=2.0.0"],
        "shap":    ["shap>=0.43.0"],
        "dev":     ["pytest>=7.4.0", "black>=23.0.0", "mypy>=1.5.0"],
    },
)
