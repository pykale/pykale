#!/usr/bin/env python3
from io import open
from os import path

from setuptools import find_packages, setup

# Core kale API dependencies
with open("requirements.txt", encoding='utf-8') as f:
    requirements = f.read().splitlines()

# Additional dependencies for examples/tutorials and development
with open("requirements-extras.txt", encoding='utf-8') as f:
    extra_requirements = f.read().splitlines()

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
    extras_require={'extras': extra_requirements},
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
