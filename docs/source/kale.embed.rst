.. kale.embed package

Embed
==================

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

``ProductOfExperts`` combines Gaussian experts using these conventions:

- **Inputs:** ``mean`` and ``log_var`` have shape ``(M, B, D)`` for experts, batch samples, and latent dimensions. The returned mean and log-variance each have shape ``(B, D)``.
- **Mask:** The optional ``mask`` has shape ``(M, B)`` and contains boolean or zero/one values; one means the expert is present. Different samples in one batch may have different masks. An absent expert receives mean zero and log-variance 20, giving negligible finite precision and zero gradient for its mean.
- **Numerics:** An all-missing sample has a finite, highly uncertain result; with ``M`` absent experts and default ``eps``, its log-variance is approximately ``20 - log(M)``. Masked float16 and bfloat16 inputs are computed in float32 and returned in their original dtype.
- **Compatibility:** No prior is added automatically. Calls without ``mask`` retain the existing tuple result and third positional ``eps`` argument.

.. code-block:: python

   import torch

   from kale.embed.multimodal_fusion import ProductOfExperts

   # mean, log_var: (M, B, D); present: (M, B) bool
   joint_mean, joint_log_var = ProductOfExperts()(mean, log_var, mask=present)

To make an all-missing sample fall back to the standard-normal prior, append an always-present prior expert:

.. code-block:: python

   prior_mean = torch.zeros_like(mean[:1])
   prior_log_var = torch.zeros_like(log_var[:1])
   prior_present = torch.ones_like(present[:1], dtype=torch.bool)
   joint_mean, joint_log_var = ProductOfExperts()(
       torch.cat((prior_mean, mean)),
       torch.cat((prior_log_var, log_var)),
       mask=torch.cat((prior_present, present)),
   )

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
