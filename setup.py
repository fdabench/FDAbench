from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "FDABench", "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "FDABench - A comprehensive database benchmark suite for evaluating DB agents"

setup(
    name="fdabench",
    version="0.1.0",
    author="FDABench Team",
    author_email="contact@fdabench.com",
    description="A comprehensive database benchmark suite for evaluating DB agents with LLM capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/fdabench/fdabench",
    packages=find_packages(where="FDABench"),
    package_dir={"": "FDABench"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "openai>=1.0.0",
        "requests>=2.28.0",
        
        # Data processing
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "duckdb>=0.8.0",
        
        # Database support (sqlite3 is built into Python)
        
        # Text processing and evaluation
        "rouge-score>=0.1.2",
        "nltk>=3.8",
        "sentence-transformers>=2.2.0",
        "sacrebleu>=2.3.0",
        
        # Utilities
        "python-dotenv>=0.19.0",
        "tqdm>=4.64.0",
        "typing-extensions>=4.0.0",
        
        # Optional web search (can be installed separately)
        "beautifulsoup4>=4.11.0",
        "lxml>=4.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "full": [
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "faiss-cpu>=1.7.0",  # For vector search
        ]
    },
    entry_points={
        "console_scripts": [
            "fdabench=FDABench.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "FDABench": [
            "prompts/*.json",
            "examples/*.py",
            "*.md",
        ],
    },
    zip_safe=False,
)