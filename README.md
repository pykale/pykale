<p align="center">
  <img src="https://github.com/pykale/pykale/raw/master/docs/images/pykale_logo.png" width="5%" alt='project-monai'> PyKale <a href="https://pypi.org/project/pykale/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pykale?color=blue"></a> <a href="https://anaconda.org/pykale/pykale"><img alt="Conda" src="https://img.shields.io/conda/v/pykale/pykale?color=blue"></a>
</p>

-----------------------------------------

![build](https://github.com/pykale/pykale/workflows/build/badge.svg)
![Unit Tests](https://github.com/pykale/pykale/workflows/unit%20tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pykale/badge/?version=latest)](https://pykale.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/personalized-badge/pykale?period=total&units=international_system&left_color=grey&right_color=lightgrey&left_text=pypi%20downloads&kill_cache=1)](https://pepy.tech/project/pykale)
![Conda](https://img.shields.io/conda/dn/pykale/pykale?color=lightgrey&label=conda%20downloads)
![GitHub all releases](https://img.shields.io/github/downloads/pykale/pykale/total?color=lightgrey&label=github%20downloads)

[Getting Started](https://github.com/pykale/pykale/tree/master/examples) |
[Documentation](https://pykale.readthedocs.io/) |
[Contributing](https://github.com/pykale/pykale/blob/master/CONTRIBUTING.md) |
[Discussions](https://github.com/pykale/pykale/discussions)

PyKale is a machine learning library that leverages knowledge from multiple sources for accurate and *interpretable* prediction. It supports graphs, images, and videos now. It is based on [PyTorch](https://pytorch.org/) and several other libraries but differs from existing ones by adopting a unified pipeline-based APIs design, enforcing standardization and minimalism, and incorporating key recent developments. See the [Trello board](https://trello.com/b/X8VBNAvf/pykale-api-overview) for an overview.

<img src="https://github.com/pykale/pykale/raw/master/docs/images/pykale_pipeline.png"
     alt="Machine learning pipeline"
     style="float: center;" />

PyKale aims to facilitate *interdisciplinary* research on *knowledge-aware* machine learning for graphs, images, and videos in computer vision, graph analysis, and medical imaging applications. Key machine learning areas of interests include **dimensionality reduction**, **deep learning**, **multimodal learning**, and **transfer learning**.

### Pipeline-based modules (core, generic, and reusable)

- `loaddata` load data from disk or online resources as in input
- `prepdata` preprocess data to fit machine learning modules below (transforms)
- `embed` embed data in a new space to learn a new representation (feature extraction/selection)
- `predict` predict a desired output
- `evaluate` evaluate the performance using some metrics
- `interpret` interpret the features and outputs via post-prediction analysis mainly via visualisation
- `pipeline` specify a machine learning workflow by combining several other modules

### Dataset-specific modules

- `examples`: Real-application on particular datasets.

## Installation

**Requirements**:
- Python >= 3.6
- PyTorch >= 1.7

Install PyKale using `pip` or `conda`:

```bash
pip install pykale
conda install -c pykale pykale
```

You need to first install [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) for `kale.embed.pipeline` and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) to work on graphs.

To upgrade to the latest (unstable) version, run

```bash
pip install --upgrade git+https://github.com/pykale/pykale.git
```

To run the unit tests:

```bash
python -m unittest
```

## Examples, Tutorials, and Discussions

See our numerous [**examples (and tutorials)**](https://github.com/pykale/pykale/tree/master/examples) on how to perform prediction tasks in PyKale.

Ask and answer questions over on [PyKale's GitHub Discussions tab](https://github.com/pykale/pykale/discussions).

## Contributing

We appreciate all contributions. You can contribute in three ways:

- [Star](https://docs.github.com/en/github/getting-started-with-github/saving-repositories-with-stars) and [fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) PyKale to follow its latest developments and share it with your networks.
- Use PyKale in your project and let us know any bugs (& fixes) and feature requests/suggestions via creating an [issue](https://github.com/pykale/pykale/issues).
- Contribute your code such as new features, functions, or examples via [fork and pull](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-collaborative-development-models) to become one of the [contributors](https://github.com/pykale/pykale/graphs/contributors).

See [contributing guidelines](https://github.com/pykale/pykale/blob/master/CONTRIBUTING.md) for more details. You can also reach us via <a href="mailto:pykale-group&#64;sheffield.ac.uk">email</a> if needed. The participation in this open source project is subject to [Code of Conduct](https://github.com/pykale/pykale/blob/master/CODE_OF_CONDUCT.md).

## The Team

PyKale is primarily maintained by a group of researchers at the University of Sheffield: [Haiping Lu](http://staffwww.dcs.shef.ac.uk/people/H.Lu/), [Raivo Koot](https://github.com/RaivoKoot), [Xianyuan Liu](https://github.com/XianyuanLiu), [Shuo Zhou](https://sz144.github.io/), and [Peizhen Bai](https://github.com/pz-white).

We would like to thank our other contributors including (but not limited to) Cameron Mcwilliam, Robert Turner, David Jones, and Will Furnass.

## Citation

    @Misc{pykale2021,
      author =   {Haiping Lu and Raivo Koot and Xianyuan Liu and Shuo Zhou and Peizhen Bai},
      title =    {{PyKale}: Knowledge-aware machine learning from multiple sources in Python},
      howpublished = {\url{https://github.com/pykale/pykale}},
      year = {2021}
    }

## Acknowledgements

The development of PyKale is partially supported by the following project(s).

- Wellcome Trust Innovator Awards: Digital Technologies Ref 215799/Z/19/Z "Developing a Machine Learning Tool to Improve Prognostic and Treatment Response Assessment on Cardiac MRI Data".
