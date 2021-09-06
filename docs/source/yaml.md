# Configuration using YAML

PyKale has been designed such that users can configure machine learning models without writing any new Python code. This is achieved using a human and machine readable language called [YAML](https://en.wikipedia.org/wiki/YAML), combined with well thought out default configuration values stored using the [YACS](https://github.com/rbgirshick/yacs) Python module. This also enables more advanced users to establish their own default and add new configuration parameters with minimal coding.

This simple example of a YAML file is used by the [digits tutorial notebook](https://github.com/pykale/pykale/blob/main/examples/digits_dann_lightn/tutorial.ipynb):

```{YAML}
DAN:
  METHOD: "CDAN"

DATASET:
  NUM_REPEAT: 1
  SOURCE: "svhn"
  VAL_SPLIT_RATIO: 0.5

SOLVER:
  MIN_EPOCHS: 0
  MAX_EPOCHS: 3

OUTPUT:
  PB_FRESH: None
```

Related configuration settings are grouped together. The group headings and allowed values are stored in a [separate Python file](https://github.com/pykale/pykale/blob/main/examples/digits_dann_lightn/config.py) which many users will not need to refer to. The headings and parameters are exaplined below:

| Heading / Parameter | Meaning | Default |
| --- | --- | --- |
| **DAN** | Domain Adaptation Net | *None* |
| METHOD | Type of DAN: `CDAN`, `CDAN-E`, or `DANN` | `CDAN` |
|**DATASET** | Dataset (for training, testing and validation ) | *None* |
| NUM_REPEAT | Number of times the training and validation cycle will be run | `10` |
| SOURCE | The source dataset path list | `mnist` |
| VAL_SPLIT_RATIO | The proportion of data to include in the training set | `0.1` |
| **SOLVER** | Model training parameters | *None* |
| MIN_EPOCHS | The minimum number of training epochs | `20` |
| MAX_EPOCHS | The maximum number of training epochs | `120` |
| **OUTPUT** | Output configuration | *None* |
| PB_FRESH | Progress bar refresh option | `0` (disabled) |

The YAML file overrides certain defaults to make the process faster and clearer for demonstration purposes. Application of an example to user data will often be as simple as amending the YAML file to reference different data location, model type and number of iterations.
