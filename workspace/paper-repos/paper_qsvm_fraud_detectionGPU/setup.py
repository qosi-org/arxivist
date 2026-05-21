"""Package setup for qsvm_fraud."""
from setuptools import setup, find_packages

setup(
    name="qsvm_fraud",
    version="0.1.0",
    description=(
        "Reproducible implementation of 'Quantum Support Vector Machine for "
        "Fraud Detection' (Ren & Zhang, IEEE CCPQT 2025)"
    ),
    author="ArXivist",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "qiskit>=1.0.0,<2.0.0",
        "qiskit-machine-learning>=0.7.0,<1.0.0",
        "qiskit-aer>=0.13.0,<1.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "imbalanced-learn>=0.11.0,<1.0.0",
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "seaborn>=0.12.0,<1.0.0",
        "joblib>=1.3.0,<2.0.0",
        "PyYAML>=6.0,<7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "black>=23.0",
            "ruff>=0.1.0",
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ]
    },
)
