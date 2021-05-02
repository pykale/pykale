# Examples

Get started with these examples that demonstrate key functionalities of PyKale.

Naming convention: `data_method` or `data_method_lightn` for lightning (optional)

## Separate data and code

All data in examples are from the public domain so they will be downloaded into local directories, either automatically or following instructions in specific examples. To keep this repository size small, we do not upload data here unless the size is less than 300KB. This is done by setting `.gitignore`. Data can be shared at [PyKale Data Repository](https://github.com/pykale/data) or other external locations, e.g., Google Drive.

## Examples available in three areas

* Image/video recognition
  * Image classification on [CIFAR via Transformer](https://github.com/pykale/pykale/tree/master/examples/cifar_cnntransformer), [CIFAR via IsoNet](https://github.com/pykale/pykale/tree/master/examples/cifar_isonet), [Digits via Domain Adaptation](https://github.com/pykale/pykale/tree/master/examples/digits_dann_lightn)
  * [Action recognition](https://github.com/pykale/pykale/tree/master/examples/action_dann_lightn)
    * [Video loading](https://github.com/pykale/pykale/tree/master/examples/video_loading)
* Bioinformatics/graph analysis
  * [Drug-target interaction prediction](https://github.com/pykale/pykale/tree/master/examples/bindingdb_deepdta)
  * [Polypharmacy side effect prediction](https://github.com/pykale/pykale/tree/master/examples/drug_gripnet)
* Medical image analysis
  * [Cardiac MRI diagnosis](https://github.com/pykale/pykale/tree/master/examples/cmri_mpca)
