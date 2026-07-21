from setuptools import find_packages, setup

setup(
    name="quantum_horizon",
    version="0.1.0",
    description="Reproduction of arXiv:2606.14484 -- Quantum Horizon (Bitcoin/Ethereum quantum threat assessment)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
