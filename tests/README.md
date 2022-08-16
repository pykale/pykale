# Test Guidelines

All code should be covered by [unit tests](https://carpentries-incubator.github.io/python-testing/04-units/index.html) with at least 70% coverage, and [regression tests](https://carpentries-incubator.github.io/python-testing/07-integration/index.html) where appropriate. When a bug is located and fixed, new test(s) should be added that would catch this bug. Please use [pykale discussions on testing](https://github.com/pykale/pykale/discussions/categories/testing) to talk about tests and ask for help. The overall and specific coverage of your commits can be checked conveniently by browsing the respective branch at the [codecov report](https://app.codecov.io/gh/pykale/pykale/commits), e.g., [check commits under the rewrite-gripnet branch](https://app.codecov.io/gh/pykale/pykale/commits?branch=rewrite-gripnet) by clicking the specific commits.

These guidelines will help you to write tests to address sufficiently compact pieces of code such that it is easy to identify causes of failure and for tests to also cover larger workflows such that confidence or trust can be built in reproducibility of outputs. We use [pytest](https://docs.pytest.org/en/stable/) (see tutorials [python testing software carpentry (alpha)](https://carpentries-incubator.github.io/python-testing/) and [tutorialspoint pytest tutorial](https://www.tutorialspoint.com/pytest/pytest_tutorial.pdf)). There is some subjectivity involved in deciding how much of the potential behaviour of your code to check.

## Quick start

- [Compact pytest tutorial](https://www.tutorialspoint.com/pytest/pytest_tutorial.pdf), [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html), [pytest exceptions](https://docs.pytest.org/en/stable/assert.html#assertions-about-expected-exceptions)
- Example [unit tests for deep learning code of variational autoencoder in PyTorch](https://github.com/tilman151/unittest_dl) and the related post [How to Trust Your Deep Learning Code](https://krokotsch.eu/cleancode/2020/08/11/Unit-Tests-for-Deep-Learning.html). *(It uses `unittest` but we use `pytest`. To convert a `unittest` to a `pytest`, [unittest2pytest](https://github.com/pytest-dev/unittest2pytest) is a good starting point.)*
- Learn from [existing pykale tests](https://github.com/pykale/pykale/tree/main/tests), [pytorch tests](https://github.com/pytorch/pytorch/tree/master/test), [torchvision tests](https://github.com/pytorch/vision/tree/master/test), and pytest+pytorch examples [fastai1 tests](https://github.com/fastai/fastai1/tree/master/tests) and [Kornia tests](https://github.com/kornia/kornia/tree/master/test)
- Use GitHub code links to find out definitions and references
- Use [Python Test Explorer for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=LittleFoxTeam.vscode-python-test-adapter) or [pytest in pycharm](https://www.jetbrains.com/help/pycharm/pytest.html) to run tests conveniently.
- [fastai testing](https://fastai1.fast.ai/dev/test.html) is a good high-level reference. We adapt its recommendations on [writing tests](https://fastai1.fast.ai/dev/test.html#writing-tests) below:
  - Think about how to create a test of the real functionality that runs quickly, e.g. based on our [`examples`](https://github.com/pykale/pykale/tree/main/examples).
  - Use module scope fixtures to run initial code that can be shared amongst tests. When using fixtures, make sure the test doesn’t modify the global object it received. If there's a risk of modifying a broadly scoped fixture, you could clone it with a more tightly scoped fixture or create a fresh fixture/object instead.
  - Avoid pretrained models, since they have to be downloaded from the internet to run the test.
  - Create some minimal data for your test, or use data already in repo’s data/ directory.

See more details below, particularly [test data](#test-data), [common parameters](#common-parameters), and [running tests locally](#running-tests-locally).

## Test data

Data needed for testing should be uploaded to [pykale/data](https://github.com/pykale/data) (preferred) or other external sources, and **automatically downloaded** via `download_file_by_url` from `kale.utils.download` during tests to `tests/test_data` as defined `download_path` of [`tests/conftest.py`](https://github.com/pykale/pykale/blob/main/tests/conftest.py). More complex test data requirements for your **pull request** can be discussed in the motivating **issue** or [pykale discussions on testing](https://github.com/pykale/pykale/discussions/categories/testing).

## Common parameters

Consider adding parameters (or objects etc.) that may be useful to multiple tests as fixtures in a [`conftest.py`](
https://docs.pytest.org/en/stable/fixture.html#conftest-py-sharing-fixtures-across-multiple-files) file, either in `tests/` or the appropriate sub-module.

## Running tests locally

To run tests locally you will need to have installed `pykale` with the development requirements:

```sh
git clone https://github.com/pykale/pykale
cd pykale
pip install -e .[dev]
```

then run:

```sh
pytest
```

## Unit tests

A **unit test** checks that a small "unit" of software (e.g. a function) performs correctly. It might, for example, check that the function `add` returns the number `2` when a list `[1, 1]` is the input.

Within the `tests/` folder is a folder structure that mimics that of the `kale` python module. Unit tests for code in a given file in `kale/` should be placed in their equivalent file in `tests/` e.g. unit tests for a function in `pykale/kale/loaddata/image_access.py` should be located in `pykale/tests/loaddata/test_image_access.py`.

Philosophically, the author of a "unit" of code knows exactly what it should do and can write the test criteria accordingly.

## Regression tests

A **regression test** checks that software produces the same results after a change is made. In `pykale`, we expect regression tests to achieve this by testing several different parts of the software at once (in effect, an [integration test](https://carpentries-incubator.github.io/python-testing/07-integration/index.html)). A single regression test might test *loading some input files*, *setting up a model* and *generating a plot* based on the model. This could be achieved by running the software with previously stored baseline inputs and checking the output is the same as previously stored baseline outputs.

Regression tests should be placed in `tests/regression`. Further subfolders can be added, as required. We plan to add regression tests covering existing functionality based on examples in the `examples/` folder.

Philosophically, regression tests treat the "past as truth" - the correct output / behaviour is the way it worked before a change.

## Testing DataFrames and arrays

Comparisons / assertions involving `pandas` `DataFrames` (or other `pandas` objects) should be made using `pandas` utility functions: [`pandas.testing.assert_frame_equal`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_frame_equal.html), [`pandas.testing.assert_series_equal`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_series_equal.html), [`pandas.testing.assert_index_equal`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_index_equal.html), [`pandas.testing.assert_extension_array_equal`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_extension_array_equal.html).

Comparisons / assertions involving `numpy` `arrays` (or other `numpy` objects) should be made using [`numpy` testing routines](https://numpy.org/doc/stable/reference/routines.testing.html). `numpy` floating point "problem" response will be [as default](https://numpy.org/doc/stable/reference/generated/numpy.seterr.html#numpy.seterr).

## Random Numbers

Random numbers in pykale are generated using base python, numpy and pytorch. Prior to making an assertion where objects that make use of random numbers are compared, the `set_seed()` function from `kale.utils.seed` should be called e.g.

In `__init__.py` or `test_<modulename>.py`:

```python
from kale.utils.seed import set_seed
```

In test, before assertion:

```python
set_seed()
```

## Logging and handling of warnings

`pytest` [captures log messages of level WARNING or above](https://docs.pytest.org/en/stable/logging.html) and outputs them to the terminal.

## Floating point errors

`numpy` can be configured to [respond differently to floating point errors](
https://numpy.org/doc/stable/reference/generated/numpy.seterr.html#numpy.seterr). `pykale` normally uses the default configuration.

## Side effects

Be aware that the code for which you are adding a test may have [side effects](https://en.wikipedia.org/wiki/Side_effect_(computer_science)) (e.g. a function changing something in a file or database, as well as returning a variable). e.g.

```python
import math

def add(numbers):
  math.pi += numbers[1] # Add first number in list to math.pi
  return sum(numbers) # Add list of numbers together

print("Sum:", add([1, 1])) # Show numbers are added
print("pi:", math.pi) # Show side effect
```

will output:

```python
Sum: 2
pi: 4.141592653589793
```

...having redefined the value of `math.pi`! `math.pi` will be redefined each time the function is run and nothing returned by the function gives any indication this has happened.

Minimising side effects makes code easier to test. Try and minimise side effects and ensure, where present, they are covered by tests.
