## pykale: machine learning for medical imaging, graph, and vision data  
We aim to develop a common framework to accelerate our research on machine learning for medical imaging, graphs/networks, 
and computer vision. If you do not have write access, please push and one from the core team will review and update.

**Coding style to follow**: [MONAI](https://github.com/Project-MONAI/MONAI) for `medim`, [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) for `graph`,
 and [kornia](https://github.com/kornia/kornia) for `vision`. All pure python from [pytorch ecosysmtem](https://pytorch.org/ecosystem/) (our goal!)  

### Objectives
* Share our resources/expertise and know each other better
* Build reusable+trustable tools for us first and the community next
* Avoid duplicated efforts and identify key missing components

### Principles
* Keep it **lean** in content, and memory/time cost. Quality first!
* Use existing top code when it fits (**credit@top + license**) and build when NA or we can do much better
* Keep it as modular as possible, following the pipeline below   

### Pipeline and modules
* `loaddata` Data loading: input
* `prepdata` Data preprocessing: transforms
* `replrnpred` Learning and prediction (joint or separate)
    * `replearn` Representation learning / embedding (feature extraction/selection)
    * `predict` Prediction (output)
* `evaluate` Evaluation: metrics
* `postanaly` Post analysis: visualisation, interpretation
* `system` Systems: system-level integration

`examples`: Demos to show functionality via notebooks or GUI applications

### Medical imaging specific
* Data and tasks
    * Brain fMRI for diagnosis, neural decoding
    * Cardiac MRI (CMRI) for diagnosis, prognosis
    * CMRI Landmark localisation
    * CMRI segmentation?
* Recommended package(s)
    * [MONAI](https://github.com/Project-MONAI/MONAI): deep learning-based healthcare imaging workflows

### Graph/Network specific
* Data and tasks
    * Knowledge graph and user-item data for recommender systems
    * Biomedical knowledge graph for drug-drug interaction prediction
* Recommended package(s)
    * [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric): deep learning library for graphs
    * [OGB](https://github.com/snap-stanford/ogb): Benchmark datasets, data loaders, and evaluators for graph machine learning    

### Computer vision specific
* Data and tasks
    * Action recognition from videos
    * Pose estimation from images
    * Image recognition (baselines)
* Recommended package(s)
    * [kornia](https://github.com/kornia/kornia): Computer Vision Library for PyTorch

### General recommendation
* Python: pytorch, pycharm (IDE)
* GitHub: GitHub Desktop, [UCL guidance](https://www.ucl.ac.uk/isd/services/research-it/research-software-development-tools/support-for-ucl-researchers-to-use-github)
* Check out and contribute to the [resources](Resources.md) (and specific resources under each domain)

We welcome contributions from expertnal members. If you have a recommendation, contact Haiping to consider and invite.