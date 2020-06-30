## pykale: machine learning for medical imaging, graph, and vision data  
We aim to develop our own ecosystem to facilitate our research on machine learning for medical imaging,
graphs/networks, and computer vision. We will identify domain-specific resources by putting them under
specific folders (medimg, graph, vision) or add prefixes (mi, gn, cv).

### Objectives
* Share our resources/expertise and know each other better
* Build reusable+trustable tools for us first and the community next
* Avoid duplicated efforts and identify key missing components

### Principles
* Keep it **lean** in content, and memory/time cost. Quality first!
* Use existing top code when it fits (**credit@top + license**) and build when NA or we can do much better
* Keep it as modular as possible, following the pipeline below   

### Pipeline and modules
* Data loading: input
* Data preprocessing: transforms
* Learning and prediction (joint or separate)
    * Representation learning / embedding (feature extraction/selection)
    * Prediction (output)
* Evaluation: metrics
* Post analysis: visualisation, interpretation
* Systems: system-level integration
* Demo: notebook and GUI 

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
    * [OGB](https://github.com/snap-stanford/ogb): 
    * PyG and DGL?

### Computer vision specific
* Data and tasks
    * Actiion recognition from videos
    * Pose estimation from images
    * Image recognition (baselines)
* Recommended package(s)
    * [?](?): 

### General recommendation
* Python: pytorch ([ecosysmtem](https://pytorch.org/ecosystem/)), tensorboardX?, pycharm (IDE)
* GitHub: GitHub Desktop, [UCL guidance](https://www.ucl.ac.uk/isd/services/research-it/research-software-development-tools/support-for-ucl-researchers-to-use-github)
* Check out and contribute to the [resources](Resources.md) (and specific resources under each domain)

We welcome contributions from expertnal members. If you have a recommendation, contact Haiping to consider and invite.