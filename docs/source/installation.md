# Installation

## Requirements

PyKale requires Python 3.7, 3.8, or 3.9. Before installing pykale, you should
- manually [install PyTorch](https://pytorch.org/get-started/locally/) matching your hardware first,
- if you will use APIs related to graphs, you need to manually install [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) first following its [official instructions](https://github.com/rusty1s/pytorch_geometric#installation) and matching your PyTorch installation, and
- If [RDKit](https://www.rdkit.org/) will be used, you need to install it via `conda install -c conda-forge rdkit`.

## Pip install

Install PyKale using `pip` for the stable version:

```bash
pip install pykale  # for the core API only
```

## Install from source

Install from source for the latest version and/or development:

```sh
git clone https://github.com/pykale/pykale
cd pykale
pip install .  # for the core API only
pip install -e .[dev]  # editable install for developers including all dependencies and examples
```

## Installation options

PyKale provides six installation options for different user needs:

- `default`: `pip install pykale` for essential functionality
- `graph`: `pip install pykale[graph]` for graph-related functionality (e.g., [TDC](https://tdcommons.ai/))
- `image`: `pip install pykale[image]` for image-related functionality (e.g., [DICOM](https://en.wikipedia.org/wiki/DICOM))
- `example`: `pip install pykale[example]` for examples and tutorials
- `full`: `pip install pykale[full]` for all functionality, including examples and tutorials
- `dev`: `pip install pykale[dev]` for development, including all functionality, examples, and tutorials

Multiple options can be chosen by separating them with commas (without whitespace). See examples below.

```sh
pip install pykale[graph,example]
pip install pykale[graph,image]
pip install pykale[graph,image,example]
```

## Tests

For local unit tests on all `kale` API, you need to have PyTorch, PyTorch Geometric, and RDKit installed (see the top) and then run [pytest](https://pytest.org/) at the root directory:

```bash
pytest
```

You can also run pytest on individual module (see [pytest documentation](https://docs.pytest.org/en/6.2.x/)).
