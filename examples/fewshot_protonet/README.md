# Prototypical Networks for Few-shot Learning

### 1. Description

This demo implements [Prototypical Networks for Few-shot Learning](https://github.com/jakesnell/prototypical-networks) within the `PyKale` framework.

ProtoNet is for few-shot learning problems under $N$-Way $K$-Shot settings:

$N$-way: It represents the number of classes or categories in one iteration in evaluation. For instance, in a 5-way scenario, data from five classes is fed into the model in an iteration in evaluation.

$K$-shot: This is the number of support samples (referred to as "shots") from each class in both training and evaluation. A 1-shot learning task indicates only one support sample per class, while a 3-shot task has three support samples per class.


## 2. Usage

### Datasets

This data loader can be used on several few-shot learning datasets, such as mini-ImageNet ([official image list](https://drive.google.com/file/d/1iBu_Iqt49opXHSUNcTRU2WQas1WICLwQ/view)/[ready-to-download data on Kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet)), tiered-ImageNet ([official dataset-generating tool](https://github.com/yaoyao-liu/tiered-imagenet-tools)/[ready-to-download data on Kaggle](https://www.kaggle.com/datasets/arjun2000ashok/tieredimagenet)), and Omniglot ([official downloading code](https://github.com/brendenlake/omniglot)/[ready-to-download data on Kaggle](https://www.kaggle.com/datasets/watesoyan/omniglot)), etc. All datasets should be organized as follows.

<pre>
└── root
    ├── train
    |   ├── class 1
    |   |   ├── image 1
    |   |   ├── image 2
    |   |   └── ...
    |   ├── class 2
    |   |   ├── image 1
    |   |   ├── image 2
    |   |   └── ...
    |   └── ...
    ├── val
    |   ├── class m
    |   |   ├── image 1
    |   |   ├── image 2
    |   |   └──...
    |   ├── class m+1
    |   |   ├── image 1
    |   |   ├── image 2
    |   |   └──...
    |   └── ...
    └── test
        ├── class n
        |   ├── image 1
        |   ├── image 2
        |   └── ...
        ├── class n+1
        |   ├── image 1
        |   ├── image 2
        |   └── ...
        └── ...
</pre>

### Examples

#### Single GPU training:

Example - Training ResNet18 on Omniglot under a 5-way-5-shot setting

`python main.py --cfg configs/demo.yaml --gpus 1`

Customized running

`python main.py --cfg configs/template.yaml --gpus 1`

- `demo.yaml` is a demo configuration file. Change `DATASET.ROOT` to fit your dataset directory for a quick demo running.
- `template.yaml` contains all changeable hyperparameters. It allows users to customize the model accordingly.
- Available backbones: any `resnet` structures from `torchvision.models` or `kale.embed.image_cnn`.


#### Test

Example - Testing ResNet18 on Omniglot under a 5-way-5-shot setting

`python main.py --cfg configs/demo.yaml --gpus 1 --ckpt {path to ckpt file}`

Customized running

`python main.py --cfg configs/template.yaml --gpus 1 --ckpt {path to ckpt file}`

The test hyperparameters are the same as the `VAL` section of the config file.

## 3. Related `kale` API

- `kale.loaddata.few_shot`: Dataset class for $N$-way $K$-shot problems.
- `kale.embed.image_cnn`: ResNet feature extractors.
- `kale.pipeline.fewshot_trainer`: ProtoNet trainer in `pl.LightningModule` class.
- `kale.predict.losses.proto_loss`: Compute the loss and accuracy for ProtoNet.

## Reference
[Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
```
@inproceedings{snell2017prototypical,
  title={Prototypical Networks for Few-shot Learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
 }
```
