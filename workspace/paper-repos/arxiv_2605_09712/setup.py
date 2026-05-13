from setuptools import setup, find_packages

setup(
    name="forecast_risk",
    version="0.1.0",
    description="Risk-Adjusted Forecast Evaluation Framework (arXiv: 2605.09712)",
    author="ArXivist — based on Goulet Coulombe (2026)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "lightgbm>=4.0",
        "torch>=2.0",
        "statsmodels>=0.14",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "black>=23.0",
            "ruff>=0.1",
            "jupyter>=1.0",
        ],
        "data": ["fredapi>=0.5"],
        "tabpfn": ["tabpfn>=0.1.9"],
    },
)
