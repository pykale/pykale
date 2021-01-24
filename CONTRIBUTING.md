# To contribute

- [Fork](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo) pykale (also see the [guide on forking projects](https://guides.github.com/activities/forking/)).
- Make changes to the source in your fork following [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).
- **Document** the update in [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and [update `docs`](https://github.com/pykale/pykale/tree/master/docs) to verify its API [documentations](https://pykale.readthedocs.io/en/latest/).
- Keep **data** and other large files [local/external](https://github.com/pykale/pykale/tree/master/examples/data) to keep the repository small (via `.gitignore`)
- Run linting via `flake8 ` to fix any style problems identified if you did not turn automatic linting on.
- Create a [pull request](https://github.com/pykale/pykale/pulls) explaining the changes and choose reviewers.
- After passing the review, your pull request gets merged and pykale has your contribution incorporated.

**Long term goal**: Satisfy the [requirements](https://pytorch.org/ecosystem/join) to join the [pytorch ecosysmtem](https://pytorch.org/ecosystem/)

## Objectives (Why this library)

- Share and consolidate resources/expertise in several related areas
- Build reusable and trustable tools for research and development
- Avoid duplicated efforts and identify key missing components

## Principles

- Keep it **lean** in content, and memory/time cost. Quality first!
- Use existing top code when it fits (**credit@top + license**) and build when NA or we can do much better
- Keep it modular following the pipeline below and separate [core functionalities](https://github.com/pykale/pykale/tree/master/kale) from [specific applications](https://github.com/pykale/pykale/tree/master/examples).

## Contributing code

Please follow the [fork and pull model](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-collaborative-development-models). Keeping the forked master clean up to date with the `pykale:master`. Create a branch on your fork to make changes and when ready, create a pull request to `pykale:master`.

## Coding

We need to design the core modules to be generic, reusable, customizable, and not specific to a particular dataset. 

### Linting

Use our [pre-commit-config.yaml](https://github.com/pykale/pykale/blob/master/.pre-commit-config.yaml) to lint your code before commit.
We mainly learned from [**GPyTorch**](https://github.com/cornellius-gp/gpytorch), [Kornia](https://github.com/kornia/kornia), [MONAI](https://github.com/Project-MONAI/MONAI), and [Torchio](https://github.com/fepegar/torchio).

You need to run linting locally and remove the flagged warnings before a push to the repo anyway.

### flake8

To install it: `pip install flake8`.

Usage:
`flake8 {source_file_or_directory}`

To get statics also
`flake8 {source_file_or_directory} --statistics`

Auto run before any commit
`pip install pre-commit`
`pre-commit install`

### Automation

We have adopted the following GitHub automation

- [Automerge](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/automatically-merging-a-pull-request): merges automatically when all reviews are completed and checks are passed.
- [Auto branch deletion](https://github.blog/changelog/2019-07-31-automatically-delete-head-branches-of-pull-requests/): deletes the head branches automatically after pull requests are merged. Deleted branches can be restored if needed.
- [Project board automation](https://docs.github.com/en/github/managing-your-work-on-github/about-automation-for-project-boards): automates project board card management.

### Coding style

- Follow [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
- Include detailed docstrings in code for generating documentations, following the [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Configure learning systems using [YAML](https://en.wikipedia.org/wiki/YAML) following [YACS](https://github.com/rbgirshick/yacs). Example: [ISONet](https://github.com/HaozhiQi/ISONet)
- Use [PyTorch](https://pytorch.org/tutorials/) when possible. **Highly recommend** [PyTorch Lightning](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09) ([Video](https://www.youtube.com/watch?v=QHww1JH7IDU))
- Key references include [MONAI](https://github.com/Project-MONAI/MONAI) for `medim`, [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) for `graph`, and [kornia](https://github.com/kornia/kornia) for `vision`.

### General recommendation

- Python: pytorch, [Visual Studio Code](https://code.visualstudio.com/download), pycharm
- GitHub: GitHub Desktop, [GitHub guides](https://guides.github.com/), [UCL guidance](https://www.ucl.ac.uk/isd/services/research-it/research-software-development-tools/support-for-ucl-researchers-to-use-github)

## Domain specifics

### Medical imaging

- Data and tasks
  - Brain fMRI for diagnosis, neural decoding ([Data](https://github.com/cMadan/openMorph))
  - Cardiac MRI (CMRI) for diagnosis, prognosis ([Data](http://www.cardiacatlas.org/challenges/))
  - CMRI Landmark localisation
  - CMRI segmentation?
  - Data: [Medical Data for Machine Learning](https://github.com/beamandrew/medical-data)
- Recommended package
  - [MONAI](https://github.com/Project-MONAI/MONAI): deep learning-based healthcare imaging workflows, with great [highlights](https://docs.monai.io/en/latest/highlights.html)

### Graph analysis

- Data and tasks
  - [Knowledge graph](https://github.com/shaoxiongji/awesome-knowledge-graph) and user-item interactions for recommender systems
  - Biomedical knowledge graph for drug-drug interaction prediction
  - Data: [OGB](https://github.com/snap-stanford/ogb), [OpenBioLink](https://github.com/OpenBioLink/OpenBioLink), [Chemistry/Biology graphs](https://github.com/mufeili/DL4MolecularGraph#benchmark-and-dataset)
- Recommended package
  - [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric): deep learning library for graphs

### Computer vision

- Data and tasks
  - Action recognition from [videos](https://www.di.ens.fr/~miech/datasetviz/): [Data at GitHub listing](https://github.com/jinwchoi/awesome-action-recognition)
  - Pose estimation from [images](https://www.simonwenkel.com/2018/12/09/Datasets-for-human-pose-estimation.html): [Data at GitHub listing](https://github.com/cbsudux/awesome-human-pose-estimation#datasets)
  - Image classification (baselines): [CVonline Image Databases (including video etc)](http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm)
- Recommended package
  - [kornia](https://github.com/kornia/kornia): Computer Vision Library for PyTorch by the OpenCV team

## Management

### Project boards

We follow [MONAI project boards](https://github.com/Project-MONAI/MONAI/projects) to manage our project boards.

### Versions

Refer to the [Semantic Versioning](https://semver.org/) guidelines. Given a version number `MAJOR.MINOR.PATCH`, increment the:

- MAJOR version when you make incompatible API changes,
- MINOR version when you add functionality in a backwards compatible manner, and
- PATCH version when you make backwards compatible bug fixes.

Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.
