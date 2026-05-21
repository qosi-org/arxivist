from setuptools import setup, find_packages

setup(
    name="stochastic_clock_barriers",
    version="1.0.0",
    description=(
        "Semi-analytic barrier option pricing under stochastic-clock volatility models "
        "with leverage corrections. Implements arXiv:2605.06677v1 (Guillaume, 2026)."
    ),
    author="Reproduction of Guillaume (2026)",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "matplotlib>=3.7.0",
            "pandas>=2.0.0",
            "jupyter>=1.0.0",
            "tqdm>=4.65.0",
        ],
    },
)
