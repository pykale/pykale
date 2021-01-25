# Contributing to PyKale

[Light involvements (viewers/users)](#light-involvements-viewersusers) |
[*Medium involvements (contributors)*](#medium-involvements-contributors) |
[**Heavy involvements (core team members)**](#heavy-involvements-core-team-members)

[Ask questions](#ask-questions) |
[Report bugs](#bug-report) |
[Suggest improvements](#bug-report) |
[*Fork & pull*](#fork-and-pull) |
[*Coding style*](#coding-style) |
[**Review & merge PRs**](#review-and-merge-pull-requests) |
[**Release and management**](#release-and-management)

Thank you for your interest! You can contribute to the PyKale project in a wide range of ways listed above, from light to heavy involvements. You can also reach us via <a href="mailto:pykale-group&#64;sheffield.ac.uk">email</a> if needed. The participation in this open source project is subject to [Code of Conduct](https://github.com/pykale/pykale/blob/master/CODE_OF_CONDUCT.md).

## Light involvements (viewers/users)

### Ask questions

### Report bugs

### Suggest improvements

## Medium involvements (contributors)

We follow PyTorch to use **US English spelling**.

### Fork and pull

Use the [*fork and pull* model]((https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-collaborative-development-models)) to contribute code to PyKale:

- [**Fork**](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) pykale (also see the [guide on forking projects](https://guides.github.com/activities/forking/)).
  - Keep the fork master branch synced with `pykale:master` by [syncing a fork](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork).
  - Install `pre-commit` to enforce style via `pip install pre-commit` and `pre-commit install` at the root.
- [Create a branch](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-and-deleting-branches-within-your-repository) based on the *latest master* in your fork with a *descriptive* name on what you plan to do, e.g. to fix an issue, starting with the issue ticket number.
  - Make changes to this branch using detailed commit messages and following the [coding style](#coding-style) below.
  - Document the update in [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). Update `docs` following [docs update steps](https://github.com/pykale/pykale/tree/master/docs). Build `docs` via `make html` and verify locally built documentations under `docs\build\html`.
  - Build tests and do tests (not enforced yet, to be done).
- Create a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) from the task branch above to the master branch `pykale:master` explaining the changes and choose reviewers, using a [template](#pull-request-template).
  - Check the [CI/CD status of the pull request](https://github.com/pykale/pykale/actions) and fix andy reported errors.
  - After passing all CI/CD tests, your pull request is ready for [review and merge](#review-and-merge-pull-requests) to have your contribution incorporated.
  - Reviewers may discuss with you and request explanations/changes before merging.
  - You need to [address merge conflicts](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/addressing-merge-conflicts) if they arise. Resolve the conflicts locally.

#### Before pull requests: pre-commit hooks

We set up several  [`pre-commit`](https://pre-commit.com/) hooks to ensure code quality, including

- Linting tools: [flake8](https://gitlab.com/pycqa/flake8), [black](https://github.com/psf/black), and [isort](https://github.com/timothycrosley/isort)).
- Static type analysis: [mypy](https://github.com/python/mypy) (to do, not yet active)
- Other hooks as specified in [`.pre-commit-config.yaml`](https://github.com/pykale/pykale/blob/master/.pre-commit-config.yaml), such as restricting the largest file size to 300KB and forbidding tabs and"Carriage Return, Line Feed".

You need to install pre-commit and the hooks from the root directory via

```bash
pip install pre-commit
pre-commit install
```

The hooks will be triggered for each new commit so that problems can be detected early. Pre-commit hooks are configured in [`.pre-commit-config.yaml`](https://github.com/pykale/pykale/blob/master/.pre-commit-config.yaml). If your commits can not pass the above checks, you need to fix them based on the error messages. 

You can fix many reported style problems automatically by running [black](https://black.readthedocs.io/en/stable/index.html) and [isort](https://pycqa.github.io/isort/) **from the root directory** (so that the PyKale configurations are used) and then fix the remaining by manual editing. For example,

```bash
pip install black # The first time
black ./kale/embed/new_module.py # "black ." do it for all files
pip install isort # The first time
isort ./kale/embed/new_module.py # "isort ." do it for all files
```

You can also run [flake8](https://flake8.pycqa.org/en/latest/) checking yourself. For example,

```bash
pip install flake8 # The first time
flake8 ./kale/embed/new_module.py --output-file ../flake8pykale.txt # "flake8 ." do it for all files
```

**Important**: Run these commands from the root directory so that the PyKale configuration files ([`setup.cfg`](https://github.com/pykale/pykale/blob/master/setup.cfg), [`pyproject.toml`](https://github.com/pykale/pykale/blob/master/pyproject.toml), and [`.pre-commit-config.yaml`](https://github.com/pykale/pykale/blob/master/.pre-commit-config.yaml)) are used for these tools. Otherwise, the default configurations will be used, which **differ** from the PyKale configurations. 

**IDE integration**: flake8 linting can be set up for both [VSCode](https://code.visualstudio.com/docs/python/linting) and [PyCharm](https://tirinox.ru/flake8-pycharm/) but you must use [`setup.cfg`](https://github.com/pykale/pykale/blob/master/setup.cfg) to configure it.

#### Automated GitHub workflows (continuous integration)

For continuous integration (CI) and continuous deployment (CD), we use several [GitHub workflows (actions)](https://github.com/pykale/pykale/actions) that will be triggered upon a push or pull request as specified at [`pykale/.github/workflows/`](https://github.com/pykale/pykale/tree/master/.github/workflows)

- Build: install Python dependencies (set up).
- Linting: run flake8 and pre-commit
- Unit tests: simple unit tests

We will make the above more complete and rigorous, e.g. with more tests and code coverage analysis etc.

#### Pull request template

We have a pull request template. Please use it for all pull requests and mark the status of your pull requests.

- **Ready**: ready for review and merge (if no problems found). Reviewers will be assigned.
- **Work in progress**: for core team's awareness of this development (e.g. to avoid duplicated efforts) and possible feedback (e.g. to find problems early, such as linting/CI issues). Not ready to merge yet. Change it to **Ready** when ready to merge.
- **Hold**: not for attention yet.

### Coding style

We aim to design the core `kale` modules to be highly **reusable**, generic, and customizable, and follow these guidelines:

- Enforce styles using [flake8](https://gitlab.com/pycqa/flake8), [black](https://github.com/psf/black), and [isort](https://github.com/timothycrosley/isort)), using common PyKale configuration files. 
- Include detailed docstrings in code for generating documentations, following the [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Configure learning systems using [YAML](https://en.wikipedia.org/wiki/YAML) following [YACS](https://github.com/rbgirshick/yacs). See our [examples](https://github.com/pykale/pykale/tree/master/examples).
- Use [PyTorch](https://pytorch.org/tutorials/) and [PyTorch Lightning](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09) ([Video](https://www.youtube.com/watch?v=QHww1JH7IDU)) as much as possible.
- If high-quality existing code from other sources are used, add credit and license information at the top of the file. 

#### Recommended development software

- Python IDE: [Visual Studio Code](https://code.visualstudio.com/download), [PyCharm](https://www.jetbrains.com/pycharm/download/)
- GitHub: GitHub Desktop, [GitHub guides](https://guides.github.com/), [GitHub documentations](https://docs.github.com/en)

## Heavy involvements (core team members)

### Review and merge pull requests

You should be a core team member to be able to review and merge a pull request.  Please follow GitHub guidelines on how to [review changes in pull requests](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/reviewing-changes-in-pull-requests) and [incorporate changes from a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-changes-from-a-pull-request) to review and merge the pull requests. The merge is automated in this project (see [Automation](#automation)).

### Release and management

The first release is done manually and we will consider a workflow to automate it in future.

#### Versions

We follow the [Semantic Versioning](https://semver.org/) guidelines. Given a version number `MAJOR.MINOR.PATCH`, increment the:

- MAJOR version when you make incompatible API changes,
- MINOR version when you add functionality in a backwards compatible manner, and
- PATCH version when you make backwards compatible bug fixes.

Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

#### Project boards

We set up [project boards](https://github.com/pykale/pykale/projects) to manage the progress of development.

#### Automation

We have adopted the following GitHub automation

- [Automerge](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/automatically-merging-a-pull-request): merges automatically when all reviews are completed and checks are passed.
- [Auto branch deletion](https://github.blog/changelog/2019-07-31-automatically-delete-head-branches-of-pull-requests/): deletes the head branches automatically after pull requests are merged. Deleted branches can be restored if needed.
- [Project board automation](https://docs.github.com/en/github/managing-your-work-on-github/about-automation-for-project-boards): automates project board card management.

## References

The following libraries from the [PyTorch ecosystem](https://pytorch.org/ecosystem/) are good resources to learn from:

- [GPyTorch](https://github.com/cornellius-gp/gpytorch): a highly efficient and modular implementation of Gaussian Processes in PyTorch
- [MONAI](https://github.com/Project-MONAI/MONAI): deep learning-based healthcare imaging workflows.
- [Kornia](https://github.com/kornia/kornia): Computer Vision Library for PyTorch by the OpenCV team
- [PyTorch_Geometric](https://github.com/rusty1s/pytorch_geometric): deep learning library for graphs
- [Torchio](https://github.com/fepegar/torchio): medical image preprocessing and augmentation toolkit for deep learning