from setuptools import setup, find_packages

setup(
    name="catnat",
    version="0.1.0",
    description="Reproduction of 'Beyond Softmax: A Natural Parameterization for Categorical Random Variables' (Manenti & Alippi, ICML 2026)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
