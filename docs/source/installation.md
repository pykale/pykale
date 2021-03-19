# Installation

PyKale requires a Python version between 3.6 and 3.8. You should [install PyTorch](https://pytorch.org/get-started/locally/) matching your hardware first. To work on graphs, install [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) first follow its official instructions.

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
