# Test Guidelines

All new code should be covered by [unit tests](https://carpentries-incubator.github.io/python-testing/04-units/index.html), and [regression tests](https://carpentries-incubator.github.io/python-testing/07-integration/index.html) where appropriate. We will extend test coverage to existing code.

Definitions of different types of tests can be somewhat subjective. These guidelines are intended to enable `pykale` to have a high level of test coverage, for these tests to address sufficiently compact pieces of code such that it is easy to identify causes of failure and for tests to also cover larger workflows such that confidence can be built in reproducibility of outputs.

Please use [pykale discussions on testing](https://github.com/pykale/pykale/discussions/categories/testing) to talk about tests and ask for help.

Refer to the [official pytest documentation](https://docs.pytest.org/en/stable/), or less formal [python testing software carpentry (alpha)](https://carpentries-incubator.github.io/python-testing/), if needed. This will help you to write tests and help with decisions on what aspects of your code need to be tested. There is some subjectivity involved in deciding how much of the potential behaviour of your code to check.

## Test runner

`pykale` uses the `pytest` test runner. This offers a balance of functionality, ease of use and wide community support.

## Unit tests

A **unit test** checks that a small "unit" of software (e.g. a function) performs correctly. It might, for example, check that the function `add` returns the number `2` when a list `[1, 1]` is the input.

Within the `tests/` folder is a folder structure that mimics that of the `kale` python module. Unit tests for code in a given file in `kale/` should be placed in their equivalent file in `tests/` e.g. unit tests for a function in `pykale/kale/loaddata/cifar_access.py` should be located in `pykale/tests/loaddata/test_cifar_access.py`.

Philosophically, the author of a "unit" of code knows exactly what it should do and can write the test criteria accordingly.

## Regression tests

A **regression test** checks that software produces the same results after a change is made. In `pykale`, we expect regression tests to achieve this by testing several different parts of the software at once (in effect, an [integration test](https://carpentries-incubator.github.io/python-testing/07-integration/index.html)). A single regression test might test *loading some input files*, *setting up a model* and *generating a plot* based on the model. This could be achieved by running the software with previously stored baseline inputs and checking the output is the same as previously stored baseline outputs.

Regression tests should be placed in `tests/regression`. Further subfolders can be added, as required. We plan to add regression tests covering existing functionality based on examples in the `examples/` folder.

Philosophically, regression tests treat the "past as truth" - the correct output / behaviour is the way it worked before a change.

## Test data

Data needed for testing should be placed in `tests/data`. This should be limited to small text files e.g. `.csv`, `.json`, `.yml`. Binary data should be stored outside the repository and referenced e.g. using a DOI. Discuss more complex test data requirements for your **pull request** in the motivating **issue**.

## Common parameters

Consider adding parameters (or objects etc.) that may be useful to multiple tests as fixtures in a [`conftest.py`](
https://docs.pytest.org/en/stable/fixture.html#conftest-py-sharing-fixtures-across-multiple-files) file, either in `tests/` or the appropriate sub-module.

## Testing DataFrames and arrays

Comparisons / assertions involving `pandas` `DataFrames` (or other `pandas` objects) should be made using `pandas` utility functions: [`pandas.testing.assert_frame_equal`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_frame_equal.html), [`pandas.testing.assert_series_equal`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_series_equal.html), [`pandas.testing.assert_index_equal`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_index_equal.html), [`pandas.testing.assert_extension_array_equal`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_extension_array_equal.html).

Comparisons / assertions involving `numpy` `arrays` (or other `numpy` objects) should be made using [`numpy` testing routines](https://numpy.org/doc/stable/reference/routines.testing.html). `numpy` floating point "problem" response will be [as default](https://numpy.org/doc/stable/reference/generated/numpy.seterr.html#numpy.seterr).

## Random Numbers

Random numbers in pykale are generated using base python, numpy and pytorch. Prior to making an assertion where objects that make use of random numbers are compared, the `set_seed()` function from `kale.utils.seed` should be called e.g.

In `__init__.py` or `test_<modulename>.py`:

```
from kale.utils.seed import set_seed
```

In test, before assertion:

```
set_seed()
```

## Logging and handling of warnings

`pytest` [captures log messages of level WARNING or above](https://docs.pytest.org/en/stable/logging.html) and outputs them to the terminal.

## Floating point errors

`numpy` can be configured to [respond differently to floating point errors](
https://numpy.org/doc/stable/reference/generated/numpy.seterr.html#numpy.seterr). `pykale` normally uses the default configuration.

## Side effects

Be aware that the code for which you are adding a test may have [side effects](https://en.wikipedia.org/wiki/Side_effect_(computer_science)) (e.g. a function changing something in a file or database, as well as returning a variable). e.g.

```{python}
import math

def add(numbers):
  math.pi += numbers[1] # Add first number in list to math.pi
  return sum(numbers) # Add list of numbers together

print("Sum:", add([1, 1])) # Show numbers are added
print("pi:", math.pi) # Show side effect
```

will output:

```
Sum: 2
pi: 4.141592653589793
```

...having redefined the value of `math.pi`! `math.pi` will be redefined each time the function is run and nothing returned by the function gives any indication this has happened.

Minimising side effects makes code easier to test. Try and minimise side effects and ensure, where present, they are covered by tests.
