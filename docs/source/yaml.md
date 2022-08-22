# Configuration using YAML

## Why YAML?

PyKale has been designed such that users can configure machine learning models and experiments without writing any new Python code. This is achieved via a human and machine readable language called [YAML](https://en.wikipedia.org/wiki/YAML). Well thought out default configuration values are first stored using the [YACS](https://github.com/rbgirshick/yacs) Python module in a `config.py` file. Several customized configurations can then be created in respective `.yaml` files.

This also enables more advanced users to establish their own default and add new configuration parameters with minimal coding. By separating code and configuration, this approach can lead to better [reproducibility](https://en.wikipedia.org/wiki/Reproducibility).

## A simple example

The following example is a simple [YAML file `tutorial.yaml`](https://github.com/pykale/pykale/blob/main/examples/digits_dann/configs/tutorial.yaml) used by the [digits tutorial notebook](https://github.com/pykale/pykale/blob/main/examples/digits_dann/tutorial.ipynb):

```{YAML}
DAN:
  METHOD: "CDAN"

DATASET:
  NUM_REPEAT: 1
  SOURCE: "svhn"
  VALID_SPLIT_RATIO: 0.5

SOLVER:
  MIN_EPOCHS: 0
  MAX_EPOCHS: 3

OUTPUT:
  PB_FRESH: None
```

Related configuration settings are grouped together. The group headings and allowed values are stored in a [separate Python file `config.py`](https://github.com/pykale/pykale/blob/main/examples/digits_dann/config.py) which many users will not need to refer to. The headings and parameters in this example are explained below:

| Heading / Parameter | Meaning | Default |
| --- | --- | --- |
| **DAN** | Domain Adaptation Net | *None* |
| METHOD | Type of DAN: `CDAN`, `CDAN-E`, or `DANN` | `CDAN` |
|**DATASET** | Dataset (for training, testing and validation ) | *None* |
| NUM_REPEAT | Number of times the training and validation cycle will be run | `10` |
| SOURCE | The source dataset name | `mnist` |
| VALID_SPLIT_RATIO | The proportion of training data used for validation | `0.1` |
| **SOLVER** | Model training parameters | *None* |
| MIN_EPOCHS | The minimum number of training epochs | `20` |
| MAX_EPOCHS | The maximum number of training epochs | `120` |
| **OUTPUT** | Output configuration | *None* |
| PB_FRESH | Progress bar refresh option | `0` (disabled) |

The tutorial YAML file `tutorial.yaml` above overrides certain defaults in `config.py` to make the machine learning process faster and clearer for demonstration purposes.

## Customization for your applications

Application of an example to your data can be as simple as creating a new YAML file to (change the defaults to) specify your data location, and other preferred configuration customization, e.g., in the choice of models and/or the number of iterations.
