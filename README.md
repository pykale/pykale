# Knowledge-aware learning for graphs, images, & videos

<img src="docs/pykaleWorkflow.png"
     alt="Machine learning workflow"
     style="float: center;" />

PyKale aims to consolidate *interdisciplinary* research on *knowledge-aware* machine learning for graphs, images, and videos in computer vision, graph analysis, and medical imaging applications, with pipeline-based APIs. See our [public Trello board](https://trello.com/b/X8VBNAvf/pykale-api-overview) for an overview.

Please star and/or fork PyKale to follow the latest update. Welcome to join / contact us via <a href="mailto:pykale-group&#64;sheffield.ac.uk">email</a>.
* **To contribute**
  * [Fork](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo) pykale (also see the [guide on forking projects](https://guides.github.com/activities/forking/)).
  * Make changes to the source in your fork following [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).
  * **Document** the update in [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and [update `docs`](https://github.com/pykale/pykale/tree/master/docs) to verify its API [documentations](https://pykale.readthedocs.io/en/latest/).
  * Keep **data** and other large files [local/external](https://github.com/pykale/pykale/tree/master/examples/data) to keep the repository small (via `.gitignore`)
  * Create a [pull request](https://github.com/pykale/pykale/pulls) explaining the changes and choose **reviewers**: Raivo, Xianyuan, Haiping, Shuo, David
  * After passing the review, your pull request gets merged and pykale has your contribution incorporated.

## Overview

**Long term goal**: Satisfy the [requirements](https://pytorch.org/ecosystem/join) to join the [pytorch ecosysmtem](https://pytorch.org/ecosystem/)

### Objectives

* Share and consolidate resources/expertise in several related areas
* Build reusable and trustable tools for research and development
* Avoid duplicated efforts and identify key missing components

### Principles

* Keep it **lean** in content, and memory/time cost. Quality first!
* Use existing top code when it fits (**credit@top + license**) and build when NA or we can do much better
* Keep it modular following the pipeline below and separate [core functionalities](https://github.com/pykale/pykale/tree/master/kale) from [specific applications](https://github.com/pykale/pykale/tree/master/examples).

### Pipeline-based modules

* `loaddata` load data from disk or online resources as in input
* `prepdata` preprocess data to fit machine learning modules below (transforms)
* `embed` embed data in a new space to learn a new representation (feature extraction/selection)
* `predict` predict a desired output
* `evaluate` evaluate the performance using some metrics
* `interpret` interpret the features and outputs via post-prediction analysis mainly via visualisation
* `pipeline` specify a machine learning workflow by combining several other modules

We need to design these core modules to be generic, reusable, customizable, and not specific to a particular dataset. 

### Dataset-specific modules

* `examples`: Real-application on particular datasets, with notebooks, GUI applications, and [TorchScript](https://pytorch.org/docs/stable/jit.html) support.

## Coding

### Coding style

* Follow [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
* Include detailed docstrings in code for generating documentations, following the [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
* Configure learning systems using [YAML](https://en.wikipedia.org/wiki/YAML) following [YACS](https://github.com/rbgirshick/yacs). Example: [ISONet](https://github.com/HaozhiQi/ISONet)
* Use [PyTorch](https://pytorch.org/tutorials/) when possible. **Highly recommend** [PyTorch Lightning](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09) ([Video](https://www.youtube.com/watch?v=QHww1JH7IDU))
* Key references include [MONAI](https://github.com/Project-MONAI/MONAI) for `medim`, [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) for `graph`, and [kornia](https://github.com/kornia/kornia) for `vision`.
* Repository structure

```
├───docs
├───examples
│   ├───cifar
│   │   ├───configs
│   │   ├───data
│   │   └───outputs
│   ├───cora
│   └───future
│       └───abidenyu
├───kale
│   ├───embed
│   ├───evaluate
│   ├───interpret
│   ├───loaddata
│   ├───pipeline
│   ├───predict
│   ├───prepdata
│   └───utils
└───tests
    └───data
```

### General recommendation

* Python: pytorch, [Visual Studio Code](https://code.visualstudio.com/download), pycharm
* GitHub: GitHub Desktop, [GitHub guides](https://guides.github.com/), [UCL guidance](https://www.ucl.ac.uk/isd/services/research-it/research-software-development-tools/support-for-ucl-researchers-to-use-github)

## Domain specifics

### Medical imaging

* Data and tasks
  * Brain fMRI for diagnosis, neural decoding ([Data](https://github.com/cMadan/openMorph))
  * Cardiac MRI (CMRI) for diagnosis, prognosis ([Data](http://www.cardiacatlas.org/challenges/))
  * CMRI Landmark localisation
  * CMRI segmentation?
  * Data: [Medical Data for Machine Learning](https://github.com/beamandrew/medical-data)
* Recommended package
  * [MONAI](https://github.com/Project-MONAI/MONAI): deep learning-based healthcare imaging workflows, with great [highlights](https://docs.monai.io/en/latest/highlights.html)

### Graph analysis

* Data and tasks
  * [Knowledge graph](https://github.com/shaoxiongji/awesome-knowledge-graph) and user-item interactions for recommender systems
  * Biomedical knowledge graph for drug-drug interaction prediction
  * Data: [OGB](https://github.com/snap-stanford/ogb), [OpenBioLink](https://github.com/OpenBioLink/OpenBioLink), [Chemistry/Biology graphs](https://github.com/mufeili/DL4MolecularGraph#benchmark-and-dataset)
* Recommended package
  * [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric): deep learning library for graphs

### Computer vision

* Data and tasks
  * Action recognition from [videos](https://www.di.ens.fr/~miech/datasetviz/): [Data at GitHub listing](https://github.com/jinwchoi/awesome-action-recognition)
  * Pose estimation from [images](https://www.simonwenkel.com/2018/12/09/Datasets-for-human-pose-estimation.html): [Data at GitHub listing](https://github.com/cbsudux/awesome-human-pose-estimation#datasets)
  * Image classification (baselines): [CVonline Image Databases (including video etc)](http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm)
* Recommended package
  * [kornia](https://github.com/kornia/kornia): Computer Vision Library for PyTorch by the OpenCV team

## Funding Acknowledgements

The development of PyKale is partially supported by the following project(s).

* Wellcome Trust Innovator Awards: Digital Technologies Ref 215799/Z/19/Z "Developing a Machine Learning Tool to Improve Prognostic and Treatment Response Assessment on Cardiac MRI Data"

[<img src="https://www.cimr.cam.ac.uk/images/Wellcomeawardlogo/image"
     alt="Wellcome Trust logo"
     width="60" height="60" />](https://wellcome.org/)
