from setuptools import setup, find_packages

setup(
    name="geomherd",
    version="0.1.0",
    description="GeomHerd: Forward-looking Herding Quantification via Ricci Flow Geometry",
    author="Lake Yang et al.",
    url="https://arxiv.org/abs/2605.11645",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "POT>=0.9.3",
        "pyyaml>=6.0",
        "scikit-learn>=1.3",
        "statsmodels>=0.14",
        "tqdm>=4.65",
    ],
    extras_require={
        "llm": ["anthropic>=0.25.0"],
        "forecasting": ["torch>=2.1.0"],
        "dev": ["pytest>=7.4", "black>=23.0", "mypy>=1.5"],
    },
)
