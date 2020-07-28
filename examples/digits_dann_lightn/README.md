# Examples in computer vision - Domain Adapatation for Image classification

This example is based on the [ADA: (Yet) Another Domain Adaptation library](https://github.com/criteo-research/pytorch-ada)

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
