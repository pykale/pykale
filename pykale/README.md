#### Organisation by a standard pipeline with modules below 
* `loaddata` Data loading functions to get the input data from somewhere.
* `prepdata` Preprocessing of data for learning and prediction, e.g. registration for images, graph construction
* `replrnpred` Machine learning functions for joint representation learning and prediction
* `replearn` Machine learning functions for representation learning, embedding, feature extraction, feature selection
* `predict` Machine learning functions for predicting an output
* `evaluate` Performance evaluation using some metrices
* `postanaly` Post-prediction analysis with visulisation, interpretation, etc.
* `system` System-level development and/or integration of the other modules

#### Name domain-specific files as so above. If this is difficult, consider three additional folders as the last resort 
* `medim` Medical image analysis with knowledge-aware machine learning
* `graph` Graph/Network analysis with knowledge-aware machine learning
* `vision` Computer vision with knowledge-aware machine learning

We also want to compile some domain-specific resources in these three folders.

 
