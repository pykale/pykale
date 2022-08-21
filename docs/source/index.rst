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
    notebooks
    yaml

.. toctree::
    :maxdepth: 1
    :caption: Kale API

    kale.loaddata
    kale.prepdata
    kale.embed
    kale.predict
    kale.evaluate
    kale.interpret
    kale.pipeline
    kale.utils

Kale APIs above are ordered following the machine learning pipeline, i.e., functionalities, rather than alphabetically.

.. toctree::
    :maxdepth: 1
    :caption: Example Projects

    Action Video - Domain Adaptation <https://github.com/pykale/pykale/tree/main/examples/action_dann>
    Autism fMRI - Domain Adaptation <https://github.com/pykale/pykale/tree/main/examples/multisite_neuroimg_adapt>
    BindingDB - DeepDTA <https://github.com/pykale/pykale/tree/main/examples/bindingdb_deepdta>
    CardiacMRI - MPCA <https://github.com/pykale/pykale/tree/main/examples/cmri_mpca>
    CIFAR - CNN Transformer <https://github.com/pykale/pykale/tree/main/examples/cifar_cnntransformer>
    CIFAR - ISONet <https://github.com/pykale/pykale/tree/main/examples/cifar_isonet>
    Digits - Domain Adaptation <https://github.com/pykale/pykale/tree/main/examples/digits_dann>
    Digits/Office - Multisource Adaptation <https://github.com/pykale/pykale/tree/main/examples/office_multisource_adapt>
    Toy Data - Domain Adaptation <https://github.com/pykale/pykale/tree/main/examples/toy_domain_adaptation>
    Polypharmacy - GripNet <https://github.com/pykale/pykale/tree/main/examples/polypharmacy_gripnet>
    Video - Data Loading <https://github.com/pykale/pykale/tree/main/examples/video_loading>

.. To study later the best way to document examples
.. examples/examples.cifar_cnntransformer
.. examples/examples.cifar_isonet
.. examples/examples.digits_dann

.. .. toctree::
..     :maxdepth: 1
..     :caption: Notebooks

.. examples/CMR_PAH.nblink
    "path": "../../examples/cmri_mpca/CMR_PAH.ipynb"

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. .. |digits_mybinder| image:: https://mybinder.org/badge_logo.svg
..     :target: https://mybinder.org/v2/gh/pykale/pykale/HEAD?filepath=examples%2Fdigits_dann%2Ftutorial.ipynb


.. .. |digits_colab| image:: https://colab.research.google.com/assets/colab-badge.svg
..     :target: https://colab.research.google.com/github/pykale/pykale/blob/main/examples/digits_dann/tutorial.ipynb
