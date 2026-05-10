from setuptools import setup, find_packages

setup(
    name="q_ising",
    version="0.1.0",
    description="Q-Ising: Dynamic Treatment on Networks (arXiv:2605.06564)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
