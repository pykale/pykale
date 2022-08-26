# Documentation

Learn more about [Python Docstrings](https://www.datacamp.com/community/tutorials/docstrings-python) to contribute high-quality documentation while coding. We follow [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

## Workflow

We call `kale` and `examples` the **root**-level modules, `kale.xxx` and `examples.xxx` the **first**-level modules, and `kale.xxx.xxx` and `examples.xxx.xxx` the **second**-level modules (and so on, if necessary).

1. **First-level module update**: If the `kale.xxx` module is not under `source` yet, run `sphinx-apidoc -o source/ ../kale` from `docs` to generate the `.rst` file under `source` first. If the `examples.xxx` module is not under `source/examples` yet, run `sphinx-apidoc -o source/examples ../examples` from `docs` to generate the `.rst` file under `source/examples` first. *Note: This should be rarely needed.*

2. **Second-level module update**: If the `kale.xxx.xxx` model is not in the documentation yet or has been *renamed*, add or revise it manually to `kale.xxx.rst` in an alphabetically-sorted position, e.g. adding the `linformer` module by inserting the following in `kale.embed.rst` right before the `mpca` module:
    ```
    kale.embed.linformer module
    ----------------------

    .. automodule:: kale.embed.linformer
    :members:
    :undoc-members:
    :show-inheritance:
    ```
    *Caution*: Alternatively (e.g., lots of modules are renamed), remove all relevant `.rst` files under `source` and then recreate them, e.g., via running `sphinx-apidoc -o source/ ../kale`. After creation, edit the heading of `.rst` files, e.g., from **kale.embed package** to **Embed** (see those in an earlier version).

3. **Final update step**: Run `make html` from `docs` to update the `.html` files under the `build` folder using the source files under the `source` folder and verify the updated documentation in a browser at `pykale/docs/build/html/index.html`. Run `make clean` will clean the `build` folder for a fresh build. Do **NOT** commit `docs/build` (see `.gitignore`). Build and view offline to check.

4. Other standardization

* Put a docstring at the top of each `.py` to summarize the module
* Docstring for a class should be at the top of the class definition, above `__init__`
* See `examples/digits_dann` and related modules for reference.

If you are aware of a better way to auto-generate documentations, create an issue or push your suggested changes.

## References

### Sphinx autodocumentation and Read the Doc

* We use [Read the Doc](https://readthedocs.org/) with [Sphinx](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/sphinx-quickstart.html).
* Documentations can be [automatically generated]([https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html)). [Tutorial: Autodocumenting your Python code with Sphinx](https://romanvm.pythonanywhere.com/post/autodocumenting-your-python-code-sphinx-part-i-5/)
* [Python Docstring Generator for VScode](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)

### Key references from [JMLR Machine Learning Open Source Software](http://www.jmlr.org/mloss/)

* [Tensor Train Decomposition on TensorFlow (T3F)](https://github.com/Bihaqo/t3f)
* [A Graph Kernel Library in Python](https://github.com/ysig/GraKeL)
* [A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning](https://github.com/scikit-learn-contrib/imbalanced-learn)
* [Deep Universal Probabilistic Programming](https://github.com/pyro-ppl/pyro)
* [A Python Toolbox for Scalable Outlier Detection](https://github.com/yzhao062/pyod)

Another documentation to learn from: the [MONAI highlights](https://docs.monai.io/en/latest/highlights.html).
