from setuptools import setup, find_packages

setup(
    name="apex_lob_trader",
    version="0.1.0",
    description="APEX Deep Double Duelling DQN for LOB Trading (arXiv:2301.08688)",
    author="ArXivist — generated from Nagy, Calliess, Zohren (2023)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0.1",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
    ],
)
