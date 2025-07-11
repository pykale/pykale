#!/usr/bin/env python3
import re
from io import open
from os import path

from setuptools import find_packages, setup

# Dependencies with options for different user needs. If updating this, you may need to update docs/requirements.txt too.
# If option names are changed, you need to update the installation guide at docs/source/installation.md respectively.
# Not all have a min-version specified, which is not uncommon. Specify when known or necessary (e.g. errors).
# The recommended practice is to install PyTorch from the official website to match the hardware first.
# To work on graphs, install torch-geometric following the official instructions at https://github.com/pyg-team/pytorch_geometric#installation

# Key reference followed: https://github.com/pyg-team/pytorch_geometric/blob/master/setup.py

# Core dependencies frequently used in PyKale Core API
install_requires = [
    "numpy>=2.0.0",  # Numpy 2.0.0+ is needed
    "pandas<=2.2.2",  # for compatibility with Colab 1.0.0 and Torch 2.3.0
    "pytorch-lightning>=2.3.2",  # in pipeline API only
    "scipy>=1.13.0",  # scipy 1.14.0 supports python 3.10+
    "scikit-learn>=1.6.1",
    "tensorly>=0.5.1",  # in factorization and model_weights API only
    "torch<=2.6.0",  # also change the version in the test.yaml when changing this next time, and update the pytorch version in the bindingdb_deepdta tutorial notebook
    "torchvision>=0.12.0",  # in download, sampler (NON-ideal), and vision API only
]

# Application-specific dependencies sorted alphabetically below

# Dependencies for graph analysis, e.g., for drug discovery using Therapeutics Data Commons (TDC)
graph_requires = [
    "networkx",
    "PyTDC<=0.3.6",
]

# Dependencies for image analysis
image_requires = [
    "glob2",
    "pydicom",
    "pylibjpeg",
    "python-gdcm",
    "scikit-image>=0.24.0",
]

# End application-specific dependencies

# Dependencies for all examples and tutorials
example_requires = [
    "ipykernel",
    "ipython",
    "matplotlib<=3.10.0",  # matplotlib 3.10.1 will cause "Building wheel for matplotlib (setup.py): finished with status 'error'" for tests
    "nilearn>=0.7.0",
    "Pillow",
    "PyTDC<=0.3.6",
    "seaborn",
    "torchsummary>=1.5.0",
    "yacs>=0.1.7",
    "pwlf",
    "wfdb",
    "xlsxwriter",
    "prettytable",
    "comet_ml",
    "neurokit2<=0.2.11",  # neurokit2 0.2.11 requires requires python>=3.10
    "captum",
    "pywavelets",
    "rdkit>=2025.3.1",  # For DrugBAN example
]

# Full dependencies except for development
full_requires = graph_requires + image_requires + example_requires

# Additional dependencies for development
dev_requires = full_requires + [
    "black==19.10b0",
    "coverage",
    "flake8",
    "flake8-print",
    "ipywidgets",
    "isort",
    "m2r",
    "mypy",
    "nbconvert",
    "nbmake>=0.8",
    "nbsphinx",
    "nbsphinx-link",
    "nbval",
    "pre-commit",
    "pytest",
    "pytest-cov",
    "recommonmark",
    "sphinx",
    "sphinx-rtd-theme",
]


# Get version
def read(*names, **kwargs):
    with open(path.join(path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open("README.md").read()
version = find_version("kale", "__init__.py")

# Run the setup
setup(
    name="pykale",
    version=version,
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
    python_requires=">=3.9,<3.12",
    install_requires=install_requires,
    extras_require={
        "graph": graph_requires,
        "image": image_requires,
        "example": example_requires,
        "full": full_requires,
        "dev": dev_requires,
    },
    license="MIT",
    keywords="machine learning, pytorch, deep learning, multimodal learning, transfer learning",
    classifiers=[
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
