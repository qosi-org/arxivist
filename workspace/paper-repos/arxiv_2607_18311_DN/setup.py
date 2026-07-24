from setuptools import find_packages, setup

setup(
    name="spr_gnn",
    version="0.1.0",
    description="Reproduction of arXiv:2607.18311 -- Approximating SPR Distance Between Phylogenetic Trees with Graph Neural Networks",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
