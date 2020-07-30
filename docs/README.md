# Documentation

Good documentation to learn from: the [MONAI highlights](https://docs.monai.io/en/latest/highlights.html).

## Sphinx autodocumentation

* Use [Sphinx](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/sphinx-quickstart.html) (already set up). 
* [https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html) to generate documentations automatically. [Tutorial: Autodocumenting your Python code with Sphinx](https://romanvm.pythonanywhere.com/post/autodocumenting-your-python-code-sphinx-part-i-5/)
* T
* [Python Docstring Generator for VScode](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)


If the package is updated, run `sphinx-apidoc -o source/ ../kale`

`sphinx-apidoc -o source/examples ../examples`

`make clean`

`make html`
