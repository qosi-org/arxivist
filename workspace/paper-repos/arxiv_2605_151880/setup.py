from setuptools import setup, find_packages

setup(
    name="futuresim",
    version="0.1.0",
    description="FutureSim: Replaying World Events to Evaluate Adaptive Agents",
    author="Goel, Chandak, Arun, Prabhu, Staab, Hardt, Andriushchenko, Geiping",
    url="https://arxiv.org/abs/2605.15188",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "lancedb>=0.6.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "sentence-transformers>=3.0.0",
        "anthropic>=0.25.0",
        "openai>=1.30.0",
        "tantivy>=0.21.0",
        "pyarrow>=14.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
        "python-dateutil>=2.9.0",
    ],
)
