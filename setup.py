"""
JamMa-FDAFT統合プロジェクトのセットアップ
"""

from setuptools import setup, find_packages
import os

# Read README
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return "JamMa-FDAFT: Ultra-lightweight Feature Matching with Joint Mamba and Double-Channel Aggregated Features"

# Read requirements
def read_requirements():
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "opencv-contrib-python>=4.5.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-image>=0.18.0",
        "matplotlib>=3.3.0",
        "einops>=0.6.0",
        "tqdm>=4.64.0",
        "tensorboard>=2.9.0",
        "wandb>=0.13.0",
        "h5py>=3.7.0",
        "kornia>=0.6.0"
    ]

setup(
    name="jamf",
    version="1.0.0",
    description="JamMa-FDAFT: Ultra-lightweight Feature Matching with Joint Mamba and Double-Channel Aggregated Features",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="JamF Team",
    author_email="jamf@example.com",
    url="https://github.com/username/jamf",
    license="MIT",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "jupyterlab>=3.4.0",
        ],
        "training": [
            "pytorch-lightning>=1.7.0",
            "hydra-core>=1.2.0",
            "omegaconf>=2.2.0",
            "timm>=0.6.0",
        ],
        "evaluation": [
            "scikit-learn>=1.1.0",
            "seaborn>=0.11.0",
            "plotly>=5.9.0",
            "pandas>=1.4.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0", 
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "jupyterlab>=3.4.0",
            "pytorch-lightning>=1.7.0",
            "hydra-core>=1.2.0",
            "omegaconf>=2.2.0",
            "timm>=0.6.0",
            "scikit-learn>=1.1.0",
            "seaborn>=0.11.0",
            "plotly>=5.9.0",
            "pandas>=1.4.0",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "jamf-train=jamf.training.trainer:main",
            "jamf-eval=jamf.training.evaluator:main",
            "jamf-demo=jamf.demo:main",
            "jamf-extract=scripts.extract_features:main",
            "jamf-match=scripts.match_images:main",
        ],
    },
    
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
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    
    keywords=[
        "computer vision",
        "feature matching", 
        "image matching",
        "mamba",
        "state space models",
        "planetary images",
        "remote sensing",
        "deep learning",
        "pytorch"
    ],
    
    include_package_data=True,
    package_data={
        "jamf": [
            "configs/*.py",
            "configs/**/*.py",
            "assets/*.yml",
            "assets/*.yaml",
            "assets/*.json",
        ]
    },
    
    project_urls={
        "Bug Reports": "https://github.com/username/jamf/issues",
        "Documentation": "https://jamf.readthedocs.io/",
        "Source": "https://github.com/username/jamf",
        "Papers": "https://jamf-papers.github.io/",
    },
)