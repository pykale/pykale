<p align="center">
  <img src="https://github.com/pykale/pykale/raw/main/docs/images/pykale_logo_long.png" width="40%" alt='project-pykale'>
</p>

> *Very cool library with lots of great ideas on moving toward 'green', efficient multimodal machine learning and AI*.

[Kevin Carlberg](https://kevintcarlberg.net/), AI Research Science Manager at Facebook Reality Labs (quoted from [tweet](https://twitter.com/kcarlberg/status/1387511298259177474)).

-----------------------------------------

[![tests](https://github.com/pykale/pykale/workflows/test/badge.svg)](https://github.com/pykale/pykale/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/pykale/badge/?version=latest)](https://pykale.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/pykale?color=blue)](https://pypi.org/project/pykale/)
[![codecov](https://codecov.io/gh/pykale/pykale/branch/main/graph/badge.svg?token=jmIYPbA2le)](https://codecov.io/gh/pykale/pykale)
[![PyPI downloads](https://pepy.tech/badge/pykale)](https://pepy.tech/project/pykale)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5557245.svg)](https://doi.org/10.5281/zenodo.5557245)

[Getting Started](https://github.com/pykale/pykale#how-to-use) |
[Documentation](https://pykale.readthedocs.io/) |
[Contributing](https://github.com/pykale/pykale/blob/main/.github/CONTRIBUTING.md) |
[Discussions](https://github.com/pykale/pykale/discussions) |
[Changelog](https://github.com/pykale/pykale/tree/main/.github/CHANGELOG.md)

PyKale is a library in the [PyTorch ecosystem](https://pytorch.org/ecosystem/) aiming to make machine learning more accessible to interdisciplinary research by bridging gaps between data, software, and end users. Both machine learning experts and end users can do better research with our accessible, scalable, and sustainable design, guided by green machine learning principles. PyKale has a unified *pipeline-based* API and focuses on [multimodal learning](https://en.wikipedia.org/wiki/Multimodal_learning) and [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) for graphs, images, texts, and videos at the moment, with supporting models on [deep learning](https://en.wikipedia.org/wiki/Deep_learning) and [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction).

PyKale enforces *standardization* and *minimalism*, via green machine learning concepts of *reducing* repetitions and redundancy, *reusing* existing resources, and *recycling* learning models across areas. PyKale will enable and accelerate *interdisciplinary*, *knowledge-aware* machine learning research for graphs, images, texts, and videos in applications including bioinformatics, graph analysis, image/video recognition, and medical imaging, with an overarching theme of leveraging knowledge from multiple sources for accurate and *interpretable* prediction.

See our [arXiv preprint](https://arxiv.org/abs/2106.09756) and four short introductory videos on YouTube: [Why build PyKale?](https://youtu.be/nybYgw-T2bM) [How was PyKale built?](https://youtu.be/jaIbkjkQvYs) [What's in PyKale?](https://youtu.be/I3vifU2rcc0) and [a 5-min summary](https://youtu.be/Snou2gg7pek).

#### Pipeline-based API

- `loaddata` loads data from disk or online resources as input
- `prepdata` preprocesses data to fit machine learning modules below (transforms)
- `embed` embeds data in a new space to learn a new representation (feature extraction/selection)
- `predict` predicts a desired output
- `evaluate` evaluates the performance using some metrics
- `interpret` interprets the features and outputs via post-prediction analysis mainly via visualization
- `pipeline` specifies a machine learning workflow by combining several other modules

#### Example usage

- `examples` demonstrate real applications on specific datasets with a standardized structure.

## How to Use

### Step 0: Installation

PyKale supports Python 3.6+. Before installing `pykale`, we suggest you to first [install PyTorch](https://pytorch.org/get-started/locally/) matching your hardware, and if graphs will be used, install [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) following its [official instructions](https://github.com/rusty1s/pytorch_geometric#installation).

Simple installation of `pykale` from [PyPI](https://pypi.org/project/pykale/):

```bash
pip install pykale
```

For more details and other options, please refer to [the installation guide](https://pykale.readthedocs.io/en/latest/installation.html).

### Step 1: Tutorials and Examples

Start with a brief [tutorial](https://pykale.readthedocs.io/en/latest/tutorial.html#usage-of-pipeline-based-api-in-examples) walking through API usage in examples or *interactive* [Jupyter notebook tutorials](https://pykale.readthedocs.io/en/latest/notebooks.html), e.g. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pykale/pykale/blob/main/examples/digits_dann_lightn/tutorial.ipynb) or  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pykale/pykale/HEAD?filepath=examples%2Fdigits_dann_lightn%2Ftutorial.ipynb) for a basic digit classification problem.

Browse through the [**examples**](https://github.com/pykale/pykale/tree/main/examples) to see the usage of PyKale in performing various prediction tasks in a wide range of applications, using a variety of settings, e.g. with or without [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).

Ask questions on [PyKale's GitHub Discussions tab](https://github.com/pykale/pykale/discussions) if you need help or create an issue if you find som

### Step 2: Building and Contributing

Build new modules and/or projects with PyKale referring to the [tutorial](https://pykale.readthedocs.io/en/latest/tutorial.html#building-new-modules-or-projects), e.g., on how to modify an existing pipeline or build a new one.

This is an open-source project welcoming your contributions. You can contribute in three ways:

- [Star](https://docs.github.com/en/github/getting-started-with-github/saving-repositories-with-stars) and [fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) PyKale to follow its latest developments, share it with your networks, and [ask questions](https://github.com/pykale/pykale/discussions)  about it.
- Use PyKale in your project and let us know any bugs (& fixes) and feature requests/suggestions via creating an [issue](https://github.com/pykale/pykale/issues).
- Contribute via [branch, fork, and pull](https://github.com/pykale/pykale/blob/main/.github/CONTRIBUTING.md#branch-fork-and-pull) for minor fixes and new features, functions, or examples to become one of the [contributors](https://github.com/pykale/pykale/graphs/contributors).

See [contributing guidelines](https://github.com/pykale/pykale/blob/main/.github/CONTRIBUTING.md) for more details. You can also reach us via <a href="mailto:pykale-group&#64;sheffield.ac.uk">email</a> if needed. The participation in this open source project is subject to [Code of Conduct](https://github.com/pykale/pykale/blob/main/.github/CODE_OF_CONDUCT.md).

## Who We Are

### The Team

PyKale is primarily maintained by a group of researchers at the University of Sheffield: [Haiping Lu](http://staffwww.dcs.shef.ac.uk/people/H.Lu/), [Raivo Koot](https://github.com/RaivoKoot), [Xianyuan Liu](https://github.com/XianyuanLiu), [Shuo Zhou](https://sz144.github.io/), [Peizhen Bai](https://github.com/pz-white), and [Robert Turner](https://github.com/bobturneruk).

We would like to thank our other contributors including (but not limited to) Cameron McWilliam, David Jones, and Will Furnass.

### Citation

```lang-latex
    @article{pykale,
      title   = {PyKale: Knowledge-Aware Machine Learning from Multiple Sources in Python},
      author  = {Haiping Lu and Xianyuan Liu and Robert Turner and Peizhen Bai and Raivo Koot and Shuo Zhou and Mustafa Chasmai and Lawrence Schobs},
      journal = {arXiv:2106.09756 [cs.LG]},
      year    = {2021}
    }
```

### Acknowledgements

The development of PyKale is partially supported by the following project(s).

- Wellcome Trust Innovator Awards: Digital Technologies Ref 215799/Z/19/Z "Developing a Machine Learning Tool to Improve Prognostic and Treatment Response Assessment on Cardiac MRI Data".
