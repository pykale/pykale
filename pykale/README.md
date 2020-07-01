# Modules

Modules are organised with the following standard pipeline

* `loaddata` Data loading functions to get the input data from somewhere.
* `prepdata` Preprocessing of data for learning and prediction, e.g. registration for images, graph construction
* `embed` Machine learning functions for embedding data into new spaces to learn a new representation, including feature extraction and feature selection
* `predict` Machine learning functions for predicting a desired output
* `evaluate` Performance evaluation using some metrices
* `interpret` Post-prediction interpretation, e.g. via visulisation, for further anlysis
* `system` System-level development that integrates the above modules to solve specific domain problems in medical imaging, graph/networks and computer vision.
