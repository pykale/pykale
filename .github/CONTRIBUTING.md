# Contributing to PyKale

[Light involvements (viewers/users)](#light-involvements-viewersusers) |
[*Medium involvements (contributors)*](#medium-involvements-contributors) |
[**Heavy involvements (maintainers)**](#heavy-involvements-maintainers)

[Ask questions](#ask-questions) |
[Report bugs](#report-bugs) |
[Suggest improvements](#suggest-improvements) |
[*Branch, fork & pull*](#branch-fork-and-pull) |
[*Coding style*](#coding-style) |
[*Test*](#testing) |
[Review & merge](#review-and-merge-pull-requests) |
[Release & management](#release-and-management)

Thank you for your interest! You can contribute to the PyKale project in a wide range of ways listed above, from light to heavy involvements. You can also reach us via <a href="mailto:pykale-group&#64;sheffield.ac.uk">email</a> if needed. The participation in this open source project is subject to [Code of Conduct](https://github.com/pykale/pykale/blob/main/CODE_OF_CONDUCT.md).

## Light involvements (viewers/users)

See the [ReadMe](https://github.com/pykale/pykale/blob/main/README.md) for installation instructions. Your contribution can start as light as asking questions.

### Ask questions

Ask any questions about PyKale on the [PyKale's GitHub Discussions tab](https://github.com/pykale/pykale/discussions) and we will discuss and answer you there. Questions help us identify *blind spots* in our development and can greatly improve the code quality.

### Report bugs

Search current issues to see whether they are already reported. If not, report bugs by [creating issues](https://github.com/pykale/pykale/issues) using the provided template. Even better, if you know how to fix them, make the suggestions and/or propose changes with [pull requests](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests).

### Suggest improvements

Suggest possible improvements such as new features or code refactoring by [creating issues](https://github.com/pykale/pykale/issues) using the respective templates. Even better, you are welcome to propose such changes with [pull requests](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests).

## Medium involvements (contributors)

We follow PyTorch to use **US English spelling** and recommend spell check via [Grazie](https://github.com/JetBrains/intellij-community/tree/master/plugins/grazie) in PyCharm and [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) in VS code with US English setting.

### Branch, fork and pull

A maintainer with *write* access can [create a branch](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-and-deleting-branches-within-your-repository) directly here in `pykale` to make changes under the [shared repository model](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-collaborative-development-models),  following the steps below while skipping the fork step.

Anyone can use the [*fork and pull* model](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-collaborative-development-models) to contribute code to PyKale:

- [**Fork**](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) pykale (also see the [guide on forking projects](https://guides.github.com/activities/forking/)).
  - Keep the fork main branch synced with `pykale:main` by [syncing a fork](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork).
  - Install `pre-commit` to enforce style via `pip install pre-commit` and `pre-commit install` at the root.
- [Create a branch](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-and-deleting-branches-within-your-repository) based on the *latest main* in your fork with a *descriptive* name on what you plan to do, e.g. to fix an issue, starting with the issue ticket number.
  - Make changes to this branch using detailed commit messages and following the [coding style](#coding-style) below. In particular, do [**frequent commits**](https://docs.github.com/en/actions/guides/about-continuous-integration#about-continuous-integration) and **small-scale pull requests** to make them more focused and easier to review.
  - [Sync your branch](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/syncing-your-branch) with the main frequently so that potential problems can be identified earlier.
  - Document the update in [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). Update `docs` following [docs update steps](https://github.com/pykale/pykale/tree/main/docs). Build `docs` via `make html` and verify locally built documentations under `docs\build\html`.
  - Build tests and do tests (not enforced yet, to be done).
- Create a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/) or [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) or from the task branch above to the main branch `pykale:main` explaining the changes, add an appropriate label or more, and choose one reviewer or more, using a [template](#pull-request-template).
  - If merged, the *title* of your PR (typically start with a *verb*) will automatically become part of the [changelog](https://github.com/pykale/pykale/blob/main/.github/CHANGELOG.md) in the next release, where the *label* of your PR will be used to group the changes into categories. Make the title and label precise and descriptive.
  - A draft pull request helps start a conversation with collaborators in a draft state. It will not be reviewed or merged until you change the status to “Ready for review” near the bottom of your pull request.
  - View the [continuous integration (CI) status checks](https://github.com/pykale/pykale/actions) to fix the found problems. Some [`test` actions](https://github.com/pykale/pykale/actions/workflows/test.yml) may fail/cancel due to server reasons, particularly `Test (macos-latest, ...)`. In such cases, [re-run the `test` workflow](https://docs.github.com/en/actions/managing-workflow-runs/re-running-a-workflow) later can usually pass. Also, when the check messages say files are changed, they mean changes in the simulated environment, *NOT* on the branch.
  - You need to [address merge conflicts](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/addressing-merge-conflicts) if they arise. Resolve the conflicts locally.
  - After passing all CI checks and resolving the conflicts, you should [request a review](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/requesting-a-pull-request-review). If you know who is appropriate or like the [suggested reviewers](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/requesting-a-pull-request-review#:~:text=Suggested%20reviewers%20are%20based%20on,review%20from%20the%20same%20reviewer.), request/assign that person. Otherwise, we will assign one shortly.
  - A reviewer will follow the [review and merge guidelines](#review-and-merge-pull-requests). The reviewer may discuss with you and request explanations/changes before merging.
  - Merging to the main branch **requires** *ALL checks to pass* AND *at least one approving review*.
  - Small pull requests are preferred for easier review. In exceptional cases of a long branch with a large number of commits in a PR, you may consider breaking it into several smaller branches and PRs, e.g. via [git-cherry-pick](https://git-scm.com/docs/git-cherry-pick), for which a [video](https://youtu.be/h8XnBRZEPYI) is available to help.

#### Before pull requests: pre-commit hooks

We set up several  [`pre-commit`](https://pre-commit.com/) hooks to ensure code quality, including

- Linting tools: [flake8](https://gitlab.com/pycqa/flake8), [black](https://github.com/psf/black), and [isort](https://github.com/timothycrosley/isort).
- Static type analysis: [mypy](https://github.com/python/mypy) (to do, not yet active)
- Other hooks as specified in [`.pre-commit-config.yaml`](https://github.com/pykale/pykale/blob/main/.pre-commit-config.yaml), such as restricting the largest file size to 300KB.

You need to install pre-commit and the hooks from the root directory via

```bash
pip install pre-commit
pre-commit install
```

This will install the `pre-commit` hooks at `pykale\.git\hooks`, to be **triggered by each new commit** to automatically run them *over the files you commit*. In this way, problems can be detected and fixed early. Several **important** points to note:

- Pre-commit hooks are configured in [`.pre-commit-config.yaml`](https://github.com/pykale/pykale/blob/main/.pre-commit-config.yaml). Only administrator should modify it.
- These hooks, e.g.,  [black](https://black.readthedocs.io/en/stable/index.html) and [isort](https://pycqa.github.io/isort/), will **automatically fix** some problems for you by **changing the files**, so please check the changes after you trigger `commit`.
- If your commits can not pass the above checks, read the error message to see what has been automatically fixed and what needs your manual fix, e.g. flake8 errors. Some flake8 errors may be fixed by some hooks so you can rerun the pre-commit (e.g. re-commit to trigger it) or just run flake8 to see the updated flake8 errors.
- If your commits can not pass the check for added large files and see the error message of `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`, try to upgrade your `git` to a version >= 2.29.2 to fix it.

#### Manual checks and fixes (be *CAREFUL*)

Required libraries will be automatically installed but if you wish, you may install them manually and run them **from the root directory** (so that the PyKale configurations are used). For example,

```bash
pip install black # The first time
black ./kale/embed/new_module.py # "black ." do it for all files
pip install isort # The first time
isort ./kale/embed/new_module.py # "isort ." do it for all files
pip install flake8 # The first time
flake8 ./kale/embed/new_module.py # "flake8 ." do it for all files
```

Run [black](https://black.readthedocs.io/en/stable/index.html) and [isort](https://pycqa.github.io/isort/) will fix the found problems automatically by modifying the files but they will be automatically run and you do *not* need to do it manually. Remaining [flake8](https://flake8.pycqa.org/en/latest/) or other errors need to be manually fixed.

**Important**: Run these commands from the root directory so that the PyKale configuration files ([`setup.cfg`](https://github.com/pykale/pykale/blob/main/setup.cfg), [`pyproject.toml`](https://github.com/pykale/pykale/blob/main/pyproject.toml), and [`.pre-commit-config.yaml`](https://github.com/pykale/pykale/blob/main/.pre-commit-config.yaml)) are used for these tools. Otherwise, the default configurations will be used, which **differ** from the PyKale configurations and are consistent.

**IDE integration**: flake8 linting can be set up for both [VSCode](https://code.visualstudio.com/docs/python/linting) and [PyCharm](https://tirinox.ru/flake8-pycharm/) but you must use [`setup.cfg`](https://github.com/pykale/pykale/blob/main/setup.cfg) to configure it. In this way, you could fix linting errors on the go.

#### Automated GitHub workflows (continuous integration)

For continuous integration (CI) and continuous deployment (CD), we use several [GitHub workflows (actions)](https://github.com/pykale/pykale/actions) that will be triggered upon a push or pull request as specified at [`pykale/.github/workflows/`](https://github.com/pykale/pykale/tree/main/.github/workflows)

- Build: install Python dependencies (set up)
- Linting: run flake8 and pre-commit
- Tests: unit and regression tests (in progress)

We will make the above more complete and rigorous, e.g. with more tests and code coverage analysis etc.

#### Pull request template

We have a pull request template. Please use it for all pull requests and mark the status of your pull requests.

- **Ready**: ready for review and merge (if no problems found). Reviewers will be assigned.
- **Work in progress**: for core team's awareness of this development (e.g. to avoid duplicated efforts) and possible feedback (e.g. to find problems early, such as linting/CI issues). Not ready to merge yet. Change it to **Ready** when ready to merge.
- **Hold**: not for attention yet.

### Coding style

We aim to design the core `kale` modules to be highly **reusable**, generic, and customizable, and follow these guidelines:

- Follow the [continuous integration practice](https://docs.github.com/en/actions/guides/about-continuous-integration#about-continuous-integration) to make small changes and commit frequently with clear descriptions for others to understand what you have done. This can detect errors sooner, reduces debug need, make it easier to merge changes, and eventually save the overall time.
- Use highly *readable* names for variables, functions, and classes. Using *verbs* is preferred when feasible for compactness. Use spell check with **US** English setting, e.g., [Grazie](https://github.com/JetBrains/intellij-community/tree/master/plugins/grazie) in PyCharm and [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) in VS code.
- Use [`logging`](https://docs.python.org/3/howto/logging.html#logging-basic-tutorial) instead of `print` to log messages. Users can choose the level via, e.g., `logging.getLogger().setLevel(logging.INFO)`. See the [benefits](https://stackoverflow.com/questions/6918493/in-python-why-use-logging-instead-of-print).
- Include detailed docstrings in code for generating documentations, following the [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- Highly reusable modules should go into `kale`. Highly data/example-specific code goes into `Examples`.
- Configure learning systems using [YAML](https://en.wikipedia.org/wiki/YAML) following [YACS](https://github.com/rbgirshick/yacs). See our [examples](https://github.com/pykale/pykale/tree/main/examples).
- Use [PyTorch](https://pytorch.org/tutorials/) and [PyTorch Lightning](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09) ([Video](https://www.youtube.com/watch?v=QHww1JH7IDU)) as much as possible.
- If high-quality existing code from other sources are used, add credit and license information at the top of the file.
- Use pre-commit hooks to enforce consistent styles via [flake8](https://gitlab.com/pycqa/flake8), [black](https://github.com/psf/black), and [isort](https://github.com/timothycrosley/isort)), with common PyKale configuration files.

#### Recommended development software

- Python IDE: [Visual Studio Code](https://code.visualstudio.com/download), [PyCharm](https://www.jetbrains.com/pycharm/download/)
- GitHub: [GitHub Desktop (for Windows/Mac)](https://desktop.github.com/), [GitKraken (for Linux)](https://www.gitkraken.com/), [GitHub guides](https://guides.github.com/), [GitHub documentations](https://docs.github.com/en)

### Testing

All new code should be covered by tests following the [`pykale` test guidelines](https://github.com/pykale/pykale/blob/main/tests/README.md).

## Heavy involvements (maintainers)

### Review and merge pull requests

A maintainer assigned to review a pull request should follow GitHub guidelines on how to [review changes in pull requests](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/reviewing-changes-in-pull-requests) and [incorporate changes from a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-changes-from-a-pull-request) to review and merge the pull requests. Merging can be automated (see [Automation](#automation)), in which case an approving review will trigger the merging. You should NOT approve the changes if they are not ready to merge.

If you think you are not the right person to review, let the administrator (haipinglu) know for a reassignment. If multiple reviewers are assigned, anyone can approve and merge unless more approvals are explicitly required.

For simple problems, such as typos, hyperlinks, the reviewers can fix it directly and push the changes rather than comment and wait for the author to fix. This will speed up the development.

### Release and management

The release will be done manually in GitHub, but with automatic upload to PyPI.

#### Versions

We follow the [Semantic Versioning](https://semver.org/) guidelines. Given a version number `MAJOR.MINOR.PATCH`, increment the:

- MAJOR version when you make incompatible API changes,
- MINOR version when you add functionality in a backwards compatible manner, and
- PATCH version when you make backwards compatible bug fixes.

Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

#### Project boards

We set up [project boards](https://github.com/pykale/pykale/projects) to manage the progress of development. A single default project contains all active/planned works, with automation.

#### Automation

We have adopted the GitHub automations including

- [Automerge](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/automatically-merging-a-pull-request): merges automatically when 1) one approving review is completed; 2) all CI checks have passed; and 3) one maintainer has **enabled the auto-merge** for a PR.
- [Auto branch deletion](https://github.blog/changelog/2019-07-31-automatically-delete-head-branches-of-pull-requests/): deletes the head branches automatically after pull requests are merged. Deleted branches can be restored if needed.
- [Project board automation](https://docs.github.com/en/github/managing-your-work-on-github/about-automation-for-project-boards): automates project board card management.

## References

The following libraries from the [PyTorch ecosystem](https://pytorch.org/ecosystem/) are good resources to learn from:

- [**PyTorchLightning**](https://github.com/PyTorchLightning/pytorch-lightning): a lightweight PyTorch wrapper for high-performance AI research
- [GPyTorch](https://github.com/cornellius-gp/gpytorch): a highly efficient and modular implementation of Gaussian processes in PyTorch
- [Kornia](https://github.com/kornia/kornia): computer vision library for PyTorch by the OpenCV team
- [MONAI](https://github.com/Project-MONAI/MONAI): deep learning-based healthcare imaging workflows
- [PyTorch_Geometric](https://github.com/rusty1s/pytorch_geometric): deep learning library for graphs
- [TensorLy](https://github.com/tensorly/tensorly): a library for tensor learning in Python
- [Torchio](https://github.com/fepegar/torchio): medical image pre-processing and augmentation toolkit for deep learning
