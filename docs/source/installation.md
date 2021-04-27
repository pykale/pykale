# Installation

## Requirement

PyKale requires a Python version between 3.6 and 3.8. You should [install PyTorch](https://pytorch.org/get-started/locally/) matching your hardware first. To work on graphs, install [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) first follow its official instructions. If [RDKit](https://www.rdkit.org/) will be used, you need to install it via `conda install -c conda-forge rdkit`.

## Pip install

Install PyKale using `pip` for the stable version:

```bash
pip install pykale  # for the core kale API only
pip install pykale[extras]  # for Examples/Tutorials (including core API)
```

## Install from source

Install from source for the latest version and/or development:

```sh
git clone https://github.com/pykale/pykale
cd pykale
pip install .  # for core API only
pip install .[extras]  # with extras for examples/tutorials
pip install -e .[dev]  # editable install for developers including all dependencies
```

## Tests

For local unit tests on all `kale` API, you need to have PyTorch, PyTorch Geometric, and RDKit installed (see the top) and then run [pytest](https://pytest.org/) at the root directory:

```bash
pytest
```

You can also run pytest on individual module (see [pytest documentation](https://docs.pytest.org/en/6.2.x/)).
