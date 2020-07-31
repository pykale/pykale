# Examples in computer vision - Domain adapatation for digit classification using PyTorch Lightning

This example is constructed by refactoring the [ADA: (Yet) Another Domain Adaptation library](https://github.com/criteo-research/pytorch-ada), with many domain adapatation algorithms included.

## Default

* Dataset: MNIST (source) to UPSP (target)
* Algorithms: DANN, CDAN, CDNA+E, ...

`python main.py --cfg configs/MN2UP-CDAN.yaml --gpus 0`

`python main.py --cfg configs/MN2UP-DANN.yaml --gpus 0`

## Notes for discussion

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

## Sample output CSV of 10 runs (generated via [CSV To Markdown Table Generator](https://donatstudios.com/CsvToMarkdownTable))

|    |                     |                     |                     |      |        |            | 
|----|---------------------|---------------------|---------------------|------|--------|------------| 
|    | source acc          | target acc          | domain acc          | seed | method | split      | 
| 0  | 0.9879999894183128  | 0.9251666567579376  | 0.4914166614034912  | 2020 | CDAN   | Validation | 
| 1  | 0.9886868991015944  | 0.9240404324664268  | 0.4774747621631832  | 2020 | CDAN   | Test       | 
| 2  | 0.9888333227427212  | 0.915999990189448   | 0.4176666621933691  | 2021 | CDAN   | Validation | 
| 3  | 0.988585888997477   | 0.9242424526746618  | 0.4765151661740674  | 2021 | CDAN   | Test       | 
| 4  | 0.9898333227320107  | 0.9173333235085008  | 0.4382499953062505  | 2022 | CDAN   | Validation | 
| 5  | 0.9884848788933596  | 0.9236363920499572  | 0.475858600497304   | 2022 | CDAN   | Test       | 
| 6  | 0.9903333227266558  | 0.9256666567525826  | 0.4074166623031488  | 2023 | CDAN   | Validation | 
| 7  | 0.988585888997477   | 0.9230303314252524  | 0.4785858733084751  | 2023 | CDAN   | Test       | 
| 8  | 0.9899999893968924  | 0.9214999901305418  | 0.3995833290537121  | 2024 | CDAN   | Validation | 
| 9  | 0.988585888997477   | 0.9237374021540744  | 0.4785858733084751  | 2024 | CDAN   | Test       | 
| 10 | 0.9903333227266558  | 0.9161666568543296  | 0.3934999957855325  | 2025 | CDAN   | Validation | 
| 11 | 0.988585888997477   | 0.9231313415293698  | 0.4771212267987721  | 2025 | CDAN   | Test       | 
| 12 | 0.9903333227266558  | 0.9168333235138562  | 0.3583333294955082  | 2026 | CDAN   | Validation | 
| 13 | 0.988585888997477   | 0.9228283112170176  | 0.4766161762781849  | 2026 | CDAN   | Test       | 
| 14 | 0.9891666560724844  | 0.9136666568811052  | 0.37308332933753263 | 2027 | CDAN   | Validation | 
| 15 | 0.9887879092057121  | 0.9235353819458396  | 0.4776262773193594  | 2027 | CDAN   | Test       | 
| 16 | 0.9894999894022476  | 0.9154999901948032  | 0.4342499953490915  | 2028 | CDAN   | Validation | 
| 17 | 0.9887879092057121  | 0.9234343718417222  | 0.4766666813302436  | 2028 | CDAN   | Test       | 
| 18 | 0.9903333227266558  | 0.9254999900877008  | 0.3904166624852223  | 2029 | CDAN   | Validation | 
| 19 | 0.9884848788933596  | 0.9228283112170176  | 0.4779798126837704  | 2029 | CDAN   | Test       | 
| 20 | 0.11333333607763052 | 0.12000000290572645 | 0.5000000121071935  | 2020 | CDAN   | Validation | 
