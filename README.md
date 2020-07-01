# Machine learning for med_img, graph, & vision

<img src="docs/pykaleWorkflow.png"
     alt="Machine learning workflow"
     style="float: center;" />

We aim to develop a common framework to accelerate (our) **interdisciplinary** research on machine learning for medical imaging, graphs/networks, and computer vision. Our framework will be complementary to existing libraries. To join this effort, clone or fork this repository and push your contributions when ready for review and merge.

## Overview

### Plan

* End Sep 2020: Internal use by all
* End Dec 2020: First public release
* Long term: Satisfy the [requirements](https://pytorch.org/ecosystem/join) to join the [pytorch ecosysmtem](https://pytorch.org/ecosystem/)

### Objectives

* Share our resources/expertise and know each other better
* Build reusable+trustable tools for us first and the community next
* Avoid duplicated efforts and identify key missing components

### Principles

* Keep it **lean** in content, and memory/time cost. Quality first!
* Use existing top code when it fits (**credit@top + license**) and build when NA or we can do much better
* Keep it as modular as possible, following the pipeline below   

### Pipeline and modules

* `loaddata` load data from disk or online resources as in input
* `prepdata` preprocess data to fit machine learning modules below (transforms)
* `embed` embed data in a new space to learn a new representation (feature extraction/selection)
* `evaluate` evaluate the performance using some metrics
* `interpret` interpret the features and outputs via post-prediction analysis mainly via visualisation
* `system` build a system using the above modules (system-level integration)

`examples`: Demos with notebooks, GUI applications, and [TorchScript](https://pytorch.org/docs/stable/jit.html) support.

## Coding

### Coding style

* Configure learning systems following [YACS](https://github.com/rbgirshick/yacs)
* Use [PyTorch](https://pytorch.org/tutorials/) when possible and follow its coding style
* Key references include [MONAI](https://github.com/Project-MONAI/MONAI) for `medim`, [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) for `graph`, and [kornia](https://github.com/kornia/kornia) for `vision`.

### General recommendation

* Python: pytorch, [Visual Studio Code](https://code.visualstudio.com/download), pycharm
* GitHub: GitHub Desktop, [UCL guidance](https://www.ucl.ac.uk/isd/services/research-it/research-software-development-tools/support-for-ucl-researchers-to-use-github)
* Check out and contribute to the [resources](Resources.md) (and specific resources under each domain)

Welcome contributions from expertnal members. If you have a recommendation, contact Haiping to consider.

## Domain specifics

### Medical imaging

* Data and tasks
  * Brain fMRI for diagnosis, neural decoding
  * Cardiac MRI (CMRI) for diagnosis, prognosis
  * CMRI Landmark localisation
  * CMRI segmentation?
* Recommended package(s)
  * [MONAI](https://github.com/Project-MONAI/MONAI): deep learning-based healthcare imaging workflows, with great [highlights](https://docs.monai.io/en/latest/highlights.html)

### Graph/Network 

* Data and tasks
  * Knowledge graph and user-item interactions for recommender systems
  * Biomedical knowledge graph for drug-drug interaction prediction
* Recommended package(s)
  * [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric): deep learning library for graphs
  * [OGB](https://github.com/snap-stanford/ogb): Benchmark datasets, data loaders, and evaluators for graph machine learning

### Computer vision

* Data and tasks
  * Action recognition from videos
  * Pose estimation from images
  * Image recognition (baselines)
* Recommended package(s)
  * [kornia](https://github.com/kornia/kornia): Computer Vision Library for PyTorch by the OpenCV team
