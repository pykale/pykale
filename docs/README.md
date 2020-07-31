# Documentation

Good documentation to learn from: the [MONAI highlights](https://docs.monai.io/en/latest/highlights.html).

## Sphinx autodocumentation and Read the Doc

* To release on [Read the Doc](https://readthedocs.org/) using [Sphinx](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/sphinx-quickstart.html) (already set up). See my test [SimplyDeep](https://simplydeep.readthedocs.io/en/latest/)
* [https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html) to generate documentations automatically. [Tutorial: Autodocumenting your Python code with Sphinx](https://romanvm.pythonanywhere.com/post/autodocumenting-your-python-code-sphinx-part-i-5/)
* [Python Docstring Generator for VScode](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)

Three key references - all from [JMLR Machine Learning Open Source Software](http://www.jmlr.org/mloss/)
* [Tensor Train Decomposition on TensorFlow (T3F)](https://github.com/Bihaqo/t3f)
* [A Graph Kernel Library in Python](https://github.com/ysig/GraKeL)
* [A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning](https://github.com/scikit-learn-contrib/imbalanced-learn)
* [Deep Universal Probabilistic Programming](https://github.com/pyro-ppl/pyro)
* [A Python Toolbox for Scalable Outlier Detection](https://github.com/yzhao062/pyod)

## Workflow

If the package is updated, run `sphinx-apidoc -o source/ ../kale` or `sphinx-apidoc -o source/examples ../examples`

`make clean`

`make html`

**Note**: `build` folder is upload for private mode sharing and will be removed (ignored) when releasing in public.
