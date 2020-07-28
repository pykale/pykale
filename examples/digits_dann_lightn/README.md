# Examples in computer vision - Domain adapatation for digit classification using PyTorch Lightning

This example is constructed by refactoring the [ADA: (Yet) Another Domain Adaptation library](https://github.com/criteo-research/pytorch-ada), with many domain adapatation algorithms included.

## Default

* Dataset: MNIST to UPSP
* Algorithms: DANN, CDAN, CDNA+E, ...

`python main.py --cfg configs/MN2UP-CDAN.yaml --gpus 0`

`python main.py --cfg configs/MN2UP-DANN.yaml --gpus 0`

## Notes

* Using PyTorch Lightning
* Removed loaddata.py, using common API in kale.loaddata instead
* kale.utils.seed: rename to set_seed? May be confusing when using multipe seeds
* The ADA code will write multiple results in a CSV, not essentail here
* cfg.OUTPUT.PB_FRESH: set to 0 at batch mode; nonzero at interactive mode
* To standardise example file structures
* What to keep here
* Top of file: docstrings for gengerating documentation later? Or comment
* Discuss im_transform in prepdata: have cifar built-in.





## PyKale examples

`digits_dann_lightn`: `lightn` stands for `lightning`. Due to the use of lightning, only three `.py` files are needed, without `loaddata` (now in `kale.loaddata`) and `trainer` (using lightning trainer). 

`main.py`: As lightning as possible

`model.py`: This has followed cnntransformer example. We may do this for the isonet example as well. 

`config.py`: The same style as cnntransformer.

## Kale core

Those starting with `da_` are for domain adapation but may be reusuable so pending further refactoring. 

`kale.embed.da_feature`: Feature extractor network
`kale.loaddata`: `digits`, `mnistm` and `usps` for loading digit datasets. `Multisource` for constructing source and target datasets for data loader. `sampler` facilitates `Multisource` dataset construction. `splits` provides common API for generating train/val/test splits, not sure whether the rename from DatasetAccess to DatasetSplit is better. 

`kale.pipeline.da_systems`: This constructs the lightning **trainers** so contains algorithms that are more pipelines rather than building modules. It may not be good to divide them into steps so they are more systems. To discuss and confirm. Shall we move other trainers in isonet or cnntransformer to here if they are standard and reusuable?

`kale.predict`: `da_classify` contains the class and domain classifiers. `losses` are various losses, which should be highly reusuable. 

`kale.prepdata.im_transform`: This is a good way to unify the interface so we consider to load cifar from here in isonet and cnntransformer.

`kale.utils.da_logger`: This logger is quite interesting and can write multiple run results to a CSV and has quite some nice handling functions that are highly reusable. Not sure whether to keep all or make it simpler. To discuss and confirm.

