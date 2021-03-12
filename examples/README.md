# Examples

Demos to show key functionalities via notebooks and GUI applications, with [TorchScript](https://pytorch.org/docs/stable/jit.html) support (in future).

Name convention: `data_method` or `data_method_lightn` for lightning

## Separate data and code

Instructions for getting data are with the specific examples. In general, we do not upload data here to keep the repository size small. All data in examples (so far) are from the public domain so they are downloaded into local directories rather than uploaded to GitHub. This is done by setting `.gitignore` (see line 7). If we do want to share data, the data should be external to the PyKale repository (Google drive is popular nowadays. More consistent ways of data sharing such as DOIs are under exploration).

## Domain-specific development in three areas (to update)

* `medim` Medical image analysis
  * Disease diagnosis on cardiac MRI (What data?)
  * Disease diagnosis on brain fMRI (ABIDE-NYU?)
  * Cardiac MRI landmark localisation (What data?)
  * Cardiac MRI segmentation (ACDC?)
  * Cardiac MRI quality analysis (What data?)
* `graph` Graph/Network analysis
  * Node classification on `Cora`
  * `biokg` subset from OGB?
  * `movielens` subset and what KG?
* `vision` Computer vision
  * Image classificaiton on `CIFAR10`
  * Action recognition from EPIC-Kitchen subset?
  * Pose estimation from COCO subset?
