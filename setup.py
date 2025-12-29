#!/usr/bin/env python
"""
AMSM Framework Setup
====================

安装脚本

Usage:
    pip install -e .
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# 读取requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="amsm-framework",
    version="1.0.0",
    author="AMSM Research Team",
    author_email="research@example.com",
    description="Adaptive Multi-Shadow Model Framework for LLM Membership Inference Attacks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/amsm_framework",
    project_urls={
        "Bug Tracker": "https://github.com/your-repo/amsm_framework/issues",
        "Documentation": "https://github.com/your-repo/amsm_framework#readme",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "full": requirements,
    },
    entry_points={
        "console_scripts": [
            "amsm-train=scripts.train:main",
            "amsm-experiment=scripts.run_experiments:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
