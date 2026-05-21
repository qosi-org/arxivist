from setuptools import setup, find_packages

setup(
    name="evolvemem",
    version="0.1.0",
    description="EVOLVEMEM: Self-Evolving Memory Architecture via AutoResearch for LLM Agents",
    author="ArXivist reproduction",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "sentence-transformers>=2.7.0",
        "rank-bm25>=0.2.2",
        "openai>=1.30.0",
        "anthropic>=0.28.0",
        "numpy>=1.26.0",
        "nltk>=3.8.1",
        "scikit-learn>=1.4.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "python-dateutil>=2.9.0",
    ],
)
