# Examples

Get started with these examples that demonstrate key functionalities of PyKale.

Naming convention: `data_method` or `problem_method`.

## Separate data and code

All data in examples are from the public domain so they will be downloaded into local directories, either automatically or following instructions in specific examples. To keep this repository size small, we do not upload data here unless the size is less than 300KB. This is done by setting `.gitignore`. Data can be shared at [PyKale Data Repository](https://github.com/pykale/data) or other external locations, e.g., Google Drive.

## Examples are available on synthetic data and three application areas

- Synthetic data analysis
  - [Toy data classification with domain adaptation](https://github.com/pykale/pykale/tree/main/examples/toy_domain_adaptation)
- Image/video recognition
  - Image classification on [CIFAR via transformers](https://github.com/pykale/pykale/tree/master/examples/cifar_cnntransformer), [CIFAR via ISONet](https://github.com/pykale/pykale/tree/master/examples/cifar_isonet), [digits via domain adaptation](https://github.com/pykale/pykale/tree/master/examples/digits_dann), [digits/office via multi-source domain adaptation](https://github.com/pykale/pykale/tree/main/examples/office_multisource_adapt)
  - [Action recognition via domain adaptation](https://github.com/pykale/pykale/tree/master/examples/action_dann)
  - [Video loading](https://github.com/pykale/pykale/tree/master/examples/video_loading)
- Bioinformatics data analysis
  - [Drug-target interaction prediction](https://github.com/pykale/pykale/tree/master/examples/bindingdb_deepdta)
  - [Polypharmacy side effect prediction](https://github.com/pykale/pykale/tree/master/examples/polypharmacy_gripnet)
- Medical image analysis
  - [Cardiac MRI diagnosis](https://github.com/pykale/pykale/tree/master/examples/cmri_mpca)
  - [Brain fMRI classification](https://github.com/pykale/pykale/tree/main/examples/multisite_neuroimg_adapt)
