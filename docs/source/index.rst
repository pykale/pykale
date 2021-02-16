.. PyKale documentation master file, created by
   sphinx-quickstart on Wed Jul 29 22:39:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


PyKale documentation
==================================

.. toctree::
    :maxdepth: 1
    :caption: Getting started

    modules

.. toctree::
    :maxdepth: 1
    :caption: Core API

    kale.loaddata
    kale.prepdata
    kale.embed
    kale.predict
    kale.pipeline
    kale.utils

The core APIs above are ordered following the machine learning pipeline rather than alphabetically.

.. toctree::
    :maxdepth: 1
    :caption: Example Projects

	CIFAR - CNN Transformer <https://github.com/pykale/pykale/tree/master/examples/cifar_cnntransformer>
	VIDEOS - Data Loading <https://github.com/pykale/pykale/tree/master/examples/video_loading>
	CIFAR - ISONet <https://github.com/pykale/pykale/tree/master/examples/cifar_isonet>
	Digits - Domain Adaptation <https://github.com/pykale/pykale/tree/master/examples/digits_dann_lightn>

.. To study later the best way to document examples
.. examples/examples.cifar_cnntransformer
.. examples/examples.cifar_isonet
.. examples/examples.digits_dann_lightn

.. toctree::
    :maxdepth: 1
    :caption: Notebooks

.. examples/CMR_PAH.nblink
    "path": "../../examples/cmri_mpca/CMR_PAH.ipynb"

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
