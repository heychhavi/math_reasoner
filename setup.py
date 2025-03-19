"""Setup script for AI-Math-Reasoning package."""

import os
from setuptools import setup, find_packages

# Read version from version.txt
with open(os.path.join("src", "ai_math_reasoning", "version.txt"), "r") as f:
    version = f.read().strip()

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ai-math-reasoning",
    version=version,
    description="A framework for enhancing mathematical reasoning capabilities in large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Abhinav Agarwal",
    author_email="your.email@example.com",
    url="https://github.com/your-username/AI-Math-Reasoning",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "ai_math_reasoning": ["version.txt"],
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "numpy>=1.24.0",
        "datasets>=2.10.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.39.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.8.0",
        "wandb>=0.15.0",
        "tensorboard>=2.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pylint>=2.15.0",
            "black>=23.0.0",
            "isort>=5.11.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-math-train=ai_math_reasoning.cli.train:main",
            "ai-math-eval=ai_math_reasoning.cli.eval:main",
            "ai-math-inference=ai_math_reasoning.cli.inference:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
