.. PyKale documentation master file, created by
   sphinx-quickstart on Wed Jul 29 22:39:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


PyKale Documentation
==================================

.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    introduction
    installation
    tutorial

.. toctree::
    :maxdepth: 1
    :caption: Kale API

    kale.loaddata
    kale.prepdata
    kale.embed
    kale.predict
    kale.pipeline
    kale.evaluate
    kale.utils

Kale APIs above are ordered following the machine learning pipeline, i.e., functionalities, rather than alphabetically.

Example Projects
################

* `Action - Domain Adaptation <https://github.com/pykale/pykale/tree/main/examples/action_dann_lightn>`_
* `BindingDB - DeepDTA <https://github.com/pykale/pykale/tree/main/examples/bindingdb_deepdta>`_
* `CardiacMRI - MPCA <https://github.com/pykale/pykale/tree/main/examples/cmri_mpca>`_
* `CIFAR - CNN Transformer <https://github.com/pykale/pykale/tree/main/examples/cifar_cnntransformer>`_
* `CIFAR - ISONet <https://github.com/pykale/pykale/tree/main/examples/cifar_isonet>`_
* `Digits - Domain Adaptation <https://github.com/pykale/pykale/tree/main/examples/digits_dann_lightn>`_ |digits_mybinder| |digits_colab|
* `Drug - GripNet <https://github.com/pykale/pykale/tree/main/examples/digits_dann_lightn>`_
* `VIDEOS - Data Loading <https://github.com/pykale/pykale/tree/main/examples/video_loading>`_

.. To study later the best way to document examples
.. examples/examples.cifar_cnntransformer
.. examples/examples.cifar_isonet
.. examples/examples.digits_dann_lightn

.. toctree::
    :maxdepth: 1
    :caption: Notebooks

.. examples/CMR_PAH.nblink
    "path": "../../examples/cmri_mpca/CMR_PAH.ipynb"

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |digits_mybinder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/pykale/pykale/HEAD?filepath=examples%2Fdigits_dann_lightn%2Fmain.ipynb


.. |digits_colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://github.com/pykale/pykale/tree/main/examples/digits_dann_lightn/main.ipynb
