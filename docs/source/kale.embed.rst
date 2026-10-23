.. kale.embed package

Embed
==================

.. toctree::
   :maxdepth: 1

   fusion_with_missing_modalities

Submodules
----------

kale.embed.attention module
--------------------------------

.. automodule:: kale.embed.attention
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.cnn module
--------------------------------

.. automodule:: kale.embed.cnn
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.factorization module
-------------------------------

.. automodule:: kale.embed.factorization
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.gcn module
---------------------

.. automodule:: kale.embed.gcn
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: message, update

kale.embed.image\_cnn module
----------------------------

.. automodule:: kale.embed.image_cnn
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.model\_lib.ban module
--------------------------------

.. automodule:: kale.embed.model_lib.drugban
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.model\_lib.gripnet module
-------------------------

.. automodule:: kale.embed.model_lib.gripnet
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: message, update

kale.embed.model\_lib.mogonet module
-------------------------

.. automodule:: kale.embed.model_lib.mogonet
    :members:
    :undoc-members:
    :show-inheritance:

kale.embed.multimodal\_encoder module
-------------------------

.. automodule:: kale.embed.multimodal_encoder
    :members:
    :undoc-members:
    :show-inheritance:

kale.embed.multimodal\_fusion module
---------------------------------

Gaussian product of experts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ProductOfExperts`` takes expert means and log-variances of shape ``(M, B, D)`` and returns a ``(B, D)`` mean/log-variance pair. Its optional ``mask`` is a boolean or zero/one ``(M, B)`` tensor: one means the expert is present for that subject. Different subjects in one batch may have different masks.

.. code-block:: python

   from kale.embed.multimodal_fusion import ProductOfExperts

   joint_mean, joint_log_var = ProductOfExperts()(mean, log_var, mask=present)

An absent expert has mean zero and log-variance 20, giving negligible finite precision and zero gradient for its mean. If every expert is absent for a subject, the output is finite and highly uncertain. The function does not add a prior: append a standard-normal expert and an always-present mask row when a model requires the prior fallback. Calls without ``mask`` retain the existing tuple result and third positional ``eps`` argument.

.. automodule:: kale.embed.multimodal_fusion
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.nn module
--------------------------------

.. automodule:: kale.embed.nn
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.signal\_cnn module
----------------------------

.. automodule:: kale.embed.signal_cnn
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.uncertainty\_fitting module
--------------------------------------

.. automodule:: kale.embed.uncertainty_fitting
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members:

kale.embed.video.feature\_extractor module
-------------------------------------------

.. automodule:: kale.embed.video.feature_extractor
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.video.i3d module
----------------------------

.. automodule:: kale.embed.video.i3d
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.video.res3d module
------------------------------

.. automodule:: kale.embed.video.res3d
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.video.se\_i3d module
--------------------------------

.. automodule:: kale.embed.video.se_i3d
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.video.se\_res3d module
----------------------------------

.. automodule:: kale.embed.video.se_res3d
   :members:
   :undoc-members:
   :show-inheritance:

kale.embed.video.selayer module
--------------------------------

.. automodule:: kale.embed.video.selayer
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: kale.embed
   :members:
   :undoc-members:
   :show-inheritance:
