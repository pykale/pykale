<p align="center">
  <img src="https://github.com/pykale/pykale/raw/main/docs/images/pykale_logo_long.png" width="40%" alt='project-pykale'>
</p>

-----------------------------------------

![tests](https://github.com/pykale/pykale/workflows/test/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pykale/badge/?version=latest)](https://pykale.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/pykale?color=blue)](https://pypi.org/project/pykale/)
[![PyPI downloads](https://pepy.tech/badge/pykale)](https://pepy.tech/project/pykale)

[Getting Started](https://github.com/pykale/pykale#how-to-use) |
[Documentation](https://pykale.readthedocs.io/) |
[Contributing](https://github.com/pykale/pykale/blob/main/.github/CONTRIBUTING.md) |
[Discussions](https://github.com/pykale/pykale/discussions) |
[Changelog](https://github.com/pykale/pykale/tree/main/.github/CHANGELOG.md)

 PyKale is a [PyTorch](https://pytorch.org/) library for [multimodal learning](https://en.wikipedia.org/wiki/Multimodal_learning) and [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) as well as [deep learning](https://en.wikipedia.org/wiki/Deep_learning) and [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) on graphs, images, texts, and videos. By adopting a unified *pipeline-based* API design, PyKale enforces *standardization* and *minimalism*, via *reusing* existing resources, *reducing* repetitions and redundancy, and *recycling* learning models across areas. PyKale aims to facilitate *interdisciplinary*, *knowledge-aware* machine learning research for graphs, images, texts, and videos in applications including bioinformatics, graph analysis, image/video recognition, and medical imaging. It focuses on leveraging knowledge from multiple sources for accurate and *interpretable* prediction. See a [12-minute introduction video on YouTube](https://youtu.be/i5BYdMfbpMQ).

#### Pipeline-based API (generic and reusable)

- `loaddata` loads data from disk or online resources as in input
- `prepdata` preprocesses data to fit machine learning modules below (transforms)
- `embed` embeds data in a new space to learn a new representation (feature extraction/selection)
- `predict` predicts a desired output
- `evaluate` evaluates the performance using some metrics
- `interpret` interprets the features and outputs via post-prediction analysis mainly via visualisation
- `pipeline` specifies a machine learning workflow by combining several other modules

#### Example usage

- `examples` demonstrate real applications on specific datasets.

## How to Use

### Step 0: Installation

PyKale supports Python 3.6+. Before installing `pykale`, we suggest you to first [install PyTorch](https://pytorch.org/get-started/locally/) matching your hardware, and if graphs will be used, install [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) following its [official instructions](https://github.com/rusty1s/pytorch_geometric#installation).

Simple installation of `pykale` from [PyPI](https://pypi.org/project/pykale/):

```bash
pip install pykale
```

For more details and other options, please refer to [the installation guide](https://pykale.readthedocs.io/en/latest/installation.html).

### Step 1: Tutorial and Examples

Start with a brief [tutorial](https://pykale.readthedocs.io/en/latest/tutorial.html#usage-of-pipeline-based-api-in-examples) walking through API usage in examples.

Browse through the [**examples**](https://github.com/pykale/pykale/tree/main/examples) to see the usage of PyKale in performing various prediction tasks in a wide range of applications, using a variety of settings, e.g. with or without [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).

Ask questions on [PyKale's GitHub Discussions tab](https://github.com/pykale/pykale/discussions) if you need help or create an issue if you find som

### Step 2: Building and Contributing

Build new modules and/or projects with PyKale referring to the [tutorial](https://pykale.readthedocs.io/en/latest/tutorial.html#building-new-modules-or-projects), e.g., on how to modify an existing pipeline or build a new one.

This is an open-source project welcoming your contributions. You can contribute in three ways:

- [Star](https://docs.github.com/en/github/getting-started-with-github/saving-repositories-with-stars) and [fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) PyKale to follow its latest developments, share it with your networks, and [ask questions](https://github.com/pykale/pykale/discussions)  about it.
- Use PyKale in your project and let us know any bugs (& fixes) and feature requests/suggestions via creating an [issue](https://github.com/pykale/pykale/issues).
- Contribute via [branch, fork, and pull](https://github.com/pykale/pykale/blob/main/CONTRIBUTING.md#branch-fork-and-pull) for minor fixes and new features, functions, or examples to become one of the [contributors](https://github.com/pykale/pykale/graphs/contributors).

See [contributing guidelines](https://github.com/pykale/pykale/blob/main/.github/CONTRIBUTING.md) for more details. You can also reach us via <a href="mailto:pykale-group&#64;sheffield.ac.uk">email</a> if needed. The participation in this open source project is subject to [Code of Conduct](https://github.com/pykale/pykale/blob/main/.github/CODE_OF_CONDUCT.md).

## Who We Are

### The Team

PyKale is primarily maintained by a group of researchers at the University of Sheffield: [Haiping Lu](http://staffwww.dcs.shef.ac.uk/people/H.Lu/), [Raivo Koot](https://github.com/RaivoKoot), [Xianyuan Liu](https://github.com/XianyuanLiu), [Shuo Zhou](https://sz144.github.io/), [Peizhen Bai](https://github.com/pz-white), and [Robert Turner](https://github.com/bobturneruk).

We would like to thank our other contributors including (but not limited to) Cameron McWilliam, David Jones, and Will Furnass.

### Citation

```lang-latex
    @Misc{pykale2021,
      author =   {Haiping Lu and Raivo Koot and Xianyuan Liu and Shuo Zhou and Peizhen Bai and Robert Turner},
      title =    {{PyKale}: Knowledge-aware machine learning from multiple sources in Python},
      howpublished = {\url{https://github.com/pykale/pykale}},
      year = {2021}
    }
```

### Acknowledgements

The development of PyKale is partially supported by the following project(s).

- Wellcome Trust Innovator Awards: Digital Technologies Ref 215799/Z/19/Z "Developing a Machine Learning Tool to Improve Prognostic and Treatment Response Assessment on Cardiac MRI Data".
