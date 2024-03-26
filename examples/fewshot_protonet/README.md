# Prototypical Networks for Few-shot Learning

### 1. Description

This demo implements [Prototypical Networks for Few-shot Learning](https://github.com/jakesnell/prototypical-networks) within the PyKale framework. The following instructions are about how to use it.

ProtoNet is specific cally engineered for few-shot learning in an $N$-Way $K$-Shot paradigm:

$N$-way: It means the number of classes or categories in one epoch in training. For instance, within a 5-way scenario, the model encounters instances emanating from five classes in an epoch.

$K$-shot: This is the number of examples (referred to as "shots") from each class in the learning process. A 1-shot learning task is that there is only one example per class, while a 3-shot task has three examples per class.

## 2. Usage

### Datasets

This data loader can be used on several few-shot learning datasets, such as [mini-ImageNet](https://www.kaggle.com/datasets/arjunashok33/miniimagenet), [tiered-ImageNet](https://www.kaggle.com/datasets/arjun2000ashok/tieredimagenet) and [Omniglot](https://github.com/brendenlake/omniglot), etc. All datasets should be organized as follows.

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
    └── test (optional)
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

Example - Training ResNet18 on Omniglot in a 5-way-5-shot Protocol

`python main.py --cfg configs/omniglot_resnet18_5way5shot.yaml --gpus 1`

Customized running

`python main.py --cfg configs/{dataset name}_{backbone}_{n}way{k}shot.yaml --gpus 1`

- Define a customized configuration file. Fill in the blank in {} in the above command.
- Available backbones: any `resnet` structures from `torchvision.models` or `kale.embed.image_cnn`.
- Example configurations can be found in `configs/omniglot_resnet18_5way5shot.yaml` and `configs/omniglot_resnet18_5way1shot.yaml`.
- Remember to change `DATASET.ROOT` item in config files to fit your dataset directory.

#### Test

Example - Testing ResNet18 on Omniglot in a 5-way-5-shot Protocol

`python main.py --cfg configs/omniglot_resnet18_5way5shot.yaml --gpus 1 --ckpt {path to ckpt file}`

Customized running

`python main.py --cfg configs/{dataset name}_{backbone}_{n}way{k}shot.yaml --gpus 1 --ckpt {path to ckpt file}`

The test hyper-parameters are the same as the `VAL` section of the config file.

If no `test` folder in the custom dataset, choose one of the following options:
- Copying & pasting `val` set and renaming it as `test`
- Changing the `mode` in defining `test_set` part in `test.py` to `val`.

## 3. Related `kale` API

- `kale.loaddata.few_shot`: Dataset class for $N$-way-$K$-shot problems.
- `kale.embed.image_cnn`: ResNet feature extractors.
- `kale.pipeline.fewshot_trainer`: ProtoNet trainer in pl.LightningModule class.
- `kale.predict.losses.proto_loss`: Compute the loss and accuracy for protonet.

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
