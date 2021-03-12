<p align="center">
  <img src="https://github.com/pykale/pykale/raw/master/docs/images/pykale_logo.png" width="5%" alt='project-monai'> PyKale
</p>

-----------------------------------------

![tests](https://github.com/pykale/pykale/workflows/test/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pykale/badge/?version=latest)](https://pykale.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/pykale?color=blue)](https://pypi.org/project/pykale/)
[![PyPI downloads](https://pepy.tech/badge/pykale)](https://pepy.tech/project/pykale)

[Getting Started](https://github.com/pykale/pykale/tree/master/examples) |
[Documentation](https://pykale.readthedocs.io/) |
[Contributing](https://github.com/pykale/pykale/blob/master/.github/CONTRIBUTING.md) |
[Discussions](https://github.com/pykale/pykale/discussions)

 PyKale is a [PyTorch](https://pytorch.org/) library for [multimodal learning](https://en.wikipedia.org/wiki/Multimodal_learning) and [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) on graphs, images, and videos. By adopting a unified *pipeline-based* API design, PyKale enforces *standardization* and *minimalism*. PyKale aims to facilitate *interdisciplinary*, *knowledge-aware* machine learning research for graphs, images, and videos in computer vision, graph analysis, and medical imaging applications. It focuses on leveraging knowledge from multiple sources for accurate and *interpretable* prediction. PyKale's other key machine learning areas of interests include **dimensionality reduction** and **deep learning**. See the [Trello board](https://trello.com/b/X8VBNAvf/pykale-api-overview) for an API overview.

### Pipeline-based core API (generic and reusable)

- `loaddata` loads data from disk or online resources as in input
- `prepdata` preprocesses data to fit machine learning modules below (transforms)
- `embed` embeds data in a new space to learn a new representation (feature extraction/selection)
- `predict` predicts a desired output
- `evaluate` evaluates the performance using some metrics
- `interpret` interprets the features and outputs via post-prediction analysis mainly via visualisation
- `pipeline` specifies a machine learning workflow by combining several other modules

### Example usage

- `examples` demonstrate real applications on specific datasets.

## Installation

You should [install PyTorch](https://pytorch.org/get-started/locally/) matching your hardware first. To work on graphs, install [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) first follow its official instructions.

Install PyKale using `pip` for the stable version:

```bash
pip install pykale  # for the core kale API only
pip install pykale[extras]  # for Examples/Tutorials (including core API)
```

Install from source for the latest version and/or development:

```sh
git clone https://github.com/pykale/pykale
cd pykale
pip install .  # for core API only
pip install .[extras]  # with extras for examples/tutorials
pip install -e .[dev]  # editable install for developers including all dependencies
```

To run the unit tests:

```bash
pytest
```

More comprehensive test cases are not yet available.

## Examples, Tutorials, and Discussions

See our numerous [**examples (and tutorials)**](https://github.com/pykale/pykale/tree/master/examples) on how to perform prediction tasks in PyKale.

Ask and answer questions over on [PyKale's GitHub Discussions tab](https://github.com/pykale/pykale/discussions).

## Contributing

We appreciate all contributions. You can contribute in three ways:

- [Star](https://docs.github.com/en/github/getting-started-with-github/saving-repositories-with-stars) and [fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) PyKale to follow its latest developments, share it with your networks, and [ask questions](https://github.com/pykale/pykale/discussions)  about it.
- Use PyKale in your project and let us know any bugs (& fixes) and feature requests/suggestions via creating an [issue](https://github.com/pykale/pykale/issues).
- Contribute via [branch, fork, and pull](https://github.com/pykale/pykale/blob/master/CONTRIBUTING.md#branch-fork-and-pull) for minor fixes and new features, functions, or examples to become one of the [contributors](https://github.com/pykale/pykale/graphs/contributors).

See [contributing guidelines](https://github.com/pykale/pykale/blob/master/.github/CONTRIBUTING.md) for more details. You can also reach us via <a href="mailto:pykale-group&#64;sheffield.ac.uk">email</a> if needed. The participation in this open source project is subject to [Code of Conduct](https://github.com/pykale/pykale/blob/master/.github/CODE_OF_CONDUCT.md).

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
