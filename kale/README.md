# Modules

We organise modules according to the following pipeline

* `loaddata` Data loading modules to get the input data from somewhere.
* `prepdata` Preprocessing of data for learning and prediction, e.g. registration for images, graph construction
* `embed` Machine learning modules for embedding data into new spaces to learn a new representation, including feature extraction and feature selection
* `predict` Machine learning modules for predicting a desired output
* `evaluate` Performance evaluation using some metrices
* `interpret` Post-prediction interpretation, e.g. via visulisation, for further anlysis
* `pipeline` Machine learning pipelines combining several other modules to solve specific domain problems in medical imaging, graph/networks and computer vision.
