#!/usr/bin/env python3

import io
import os
import re

from setuptools import find_packages, setup


# Get version
def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open("README.md").read()
version = find_version("kale", "__init__.py")


torch_min = "1.7"
install_requires = [">=".join(["torch", torch_min]), "scikit-learn"]
# if recent dev version of PyTorch is installed, no need to install stable
try:
    import torch

    if torch.__version__ >= torch_min:
        install_requires = []
except ImportError:
    pass


# Run the setup
setup(
    name="pykale",
    version=version,
    description="A Python library for knowledge-aware machine learning",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="The PyKale team",
    url="https://github.com/pykale/pykale",
    author_email="pykale-group@sheffield.ac.uk",
    project_urls={
        "Documentation": "https://pykale.readthedocs.io",
        "Source": "https://github.com/pykale/pykale",
    },
    keywords='machine learning, pytorch, multimodal learning, transfer learning',
    license="MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        "Programming Language :: Python :: 3",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Software Development :: Libraries',        
        'Natural Language :: English',
    ],    
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require={
        "dev": ["black", "twine", "pre-commit"],
        "docs": ["ipython", "ipykernel", "sphinx<3.0.0", "sphinx_rtd_theme", "nbsphinx", "m2r"],
        "examples": ["scikit-image", "tensorly", "matplotlib", "numpy", "torchvision", "tqdm", "pytorch-lightning"],
        "torchsummary": ["torchsummary>=1.5.0"],
        "yacs": ["yacs>=0.1.7"],
        "test": ["flake8", "flake8-print", "pytest", "nbval"],
    },
)