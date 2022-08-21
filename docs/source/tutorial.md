# Tutorial

For *interactive* tutorials, see [Jupyter Notebook tutorials](notebooks.md).

## Usage of Pipeline-based API in Examples

The `kale` API has a unique pipeline-based API design. Each example typically has three essential modules (`main.py`, `config.py`, `model.py`), one optional directory (`configs`), and possibly other modules (`trainer.py`):

- `main.py` is the main module to be run, showing the main workflow.
- `config.py` is the configuration module that sets up the data, prediction problem, and hyper-parameters, etc. The settings in this module is the default configuration.
  - `configs` is the directory to place *customized* configurations for individual runs. We use `.yaml` files for this purpose.
- `model.py` is the model module to define the machine learning model and configure its training parameters.
  - `trainer.py` is the trainer module to define the training and testing workflow. This module is *only needed when NOT using `PyTorch Lightning`*.

Next, we explain the usage of the pipeline-based API in the modules above, mainly using the [domain adaptation for digits classification example](https://github.com/pykale/pykale/tree/main/examples/digits_dann).

- The `kale.pipeline` module provides mature, off-the-shelf machine learning pipelines for plug-in usage, e.g. `import kale.pipeline.domain_adapter as domain_adapter` in [`digits_dann`'s `model` module](https://github.com/pykale/pykale/blob/main/examples/digits_dann/model.py).
- The `kale.utils` module provides common utility functions, such as `from kale.utils.seed import set_seed` in [`digits_dann`'s `main` module](https://github.com/pykale/pykale/blob/main/examples/digits_dann/main.py).
- The `kale.loaddata` module provides the input to the machine learning system, such as`from kale.loaddata.image_access import DigitDatase` in  [`digits_dann`'s `main` module](https://github.com/pykale/pykale/blob/main/examples/digits_dann/main.py).
- The `kale.prepdata` module provides pre-processing functions to transform the raw input data into a suitable form for machine learning, such as `import kale.prepdata.image_transform as image_transform` in `kale.loaddata.image_access` used in  [`digits_dann`'s `main` module](https://github.com/pykale/pykale/blob/main/examples/digits_dann/main.py) for image data augmentation.
- The `kale.embed` module provides *embedding* functions (the *encoder*) to *learn* suitable representations from the (pre-processed) input data, such as `from kale.embed.image_cnn import SmallCNNFeature` in [`digits_dann`'s `model` module](https://github.com/pykale/pykale/blob/main/examples/digits_dann/model.py). This is a machine learning module.
- The `kale.predict` module provides prediction functions (the *decoder*) to *learn* a mapping from the input representation to a target prediction, such as `from kale.predict.class_domain_nets import ClassNetSmallImage` in [`digits_dann`'s `model` module](https://github.com/pykale/pykale/blob/main/examples/digits_dann/model.py). This is also a machine learning module.
- The `kale.evaluate` module implements evaluation metrics not yet available, such as the Concordance Index (CI) for measuring the proportion of [concordant pairs](https://en.wikipedia.org/wiki/Concordant_pair).
- The `kale.interpret` module aims to provide functions for interpretation of the learned model or the prediction results, such as visualization. This module has no implementation yet.

## Building New Modules or Projects

New modules/projects can be built following the steps below.

- Step 1 - Examples: Choose one of the [examples](https://github.com/pykale/pykale/tree/main/examples) of your interest (e.g., most relevant to your project) to
  - browse through the configuration, main, and model modules
  - download the data if needed
  - run the example following instructions in the example's README
- Step 2a - New model: To develop new machine learning models under PyKale,
  - define the blocks in your pipeline to figure out what the methods are for data loading, pre-processing data, embedding (encoder/representation), prediction (decoder), evaluation, and interpretation (if needed)
  - modify existing pipelines with your customized blocks or build a new pipeline with PyKale blocks and blocks from other libraries
- Step 2b - New applications: To develop new applications using PyKale,
  - clarify the input data and the prediction target to find matching functionalities in PyKale (request if not found)
  - tailor data loading, pre-processing, and evaluation (and interpretation if needed) to your application

## The Scope of Support

### Data

PyKale currently supports graphs, images, and videos, using PyTorch Dataloaders wherever possible. Audios are not supported yet (welcome your contribution).

### Machine learning models

PyKale supports modules from the following areas of machine learning

- Deep learning: convolutional neural networks (CNNs), graph neural networks (GNNs) GNN including graph convolutional networks (GCNs), transformers
- Transfer learning: domain adaptation
- Multimodal learning: integration of heterogeneous data
- Dimensionality reduction: multilinear subspace learning, such as multilinear principal component analysis (MPCA)

### Example applications

PyKale includes example application from three areas below

- Image/video recognition: imaging recognition with CIFAR10/100, digits (MNIST, USPS, SVHN), action videos (EPIC Kitchen)
- Bioinformatics/graph analysis: link prediction problems in BindingDB and knowledge graphs
- Medical imaging: cardiac MRI classification
