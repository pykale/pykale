#!/usr/bin/env python3
from io import open
from os import path

from setuptools import find_packages, setup

# Not all packages have a min-version specified, which is not uncommon. Specify when needed (e.g. errors).
# Install PyTorch from the official website to match your hardware.
# To work on graphs, install torch-geometric following the official instructions (e.g. below):
# python -m pip install torch-cluster torch-scatter torch-sparse torch-spline

# Core kale API dependencies, update docs/requirements.txt too
requirements = [
    "numpy>=1.18.0",
    "scikit-learn",
    "scikit-image",
    "torch>=1.7.0",
    "torchvision>=0.8.1",
    "pytorch-lightning",
    "tensorly",
]

# Dependencies for examples/tutorials and development
extras = {
    "docs": [
        "ipython",
        "ipykernel",
        "sphinx",
        "sphinx_rtd_theme",
        "nbsphinx",
        "m2r",
    ],
    "check (linting)": [
        "black",
        "flake8",
        "flake8-print",
        "isort",
        "mypy",
        "pre-commit",
    ],
    "utils": [
        "matplotlib",
        "torchsummary>=1.5.0",
        "tqdm>=4.1.0",
        "yacs>=0.1.7",
    ],
    "test": [
        "nbval",
        "pytest",
    ],
    "publish": [
        "twine",
    ],
},

# Alternative choice but not flexible in separating essential and extra --> NOT adopted
# read the contents of requirements.txt
# with open("requirements.txt", encoding='utf-8') as f:
#     requirements = f.read().splitlines()

# Get __version__ from __init__.py in kale
version_file = path.join("kale", "__init__.py")
with open(version_file) as f:
    exec(f.read())

# Get README content
with open('README.md', encoding='utf8') as readme_file:
    readme = readme_file.read()

# Run the setup
setup(
    name="pykale",
    version=__version__,  # noqa: F821; variable available from exec(f.read())
    description="Knowledge-aware machine learning from multiple sources in Python",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="The PyKale team",
    url="https://github.com/pykale/pykale",
    author_email="pykale-group@sheffield.ac.uk",
    project_urls={
        "Bug Tracker": "https://github.com/pykale/pykale/issues",
        "Documentation": "https://pykale.readthedocs.io",
        "Source": "https://github.com/pykale/pykale",
    },
    packages=find_packages(exclude=("docs", "examples", "tests")),
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require=extras,
    setup_requires=['setuptools>=38.6.0'],
    license="MIT",
    keywords="machine learning, pytorch, deep learning, multimodal learning, transfer learning",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries",
        "Natural Language :: English",
    ],
)
