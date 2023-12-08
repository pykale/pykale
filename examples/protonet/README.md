# Prototypical Networks for Few-shot Learning

### 1. Description

This example demonstrates how to implement [Prototypical Networks for Few-shot Learning](https://github.com/jakesnell/prototypical-networks) within the PyKale pipeline using PyTorch-Lightning. Below, we offer guidance on how to use ProtoNet effectively.

ProtoNet is designed for the few-shot learning under N-Way-K-Shot setting. 

- N-way: This refers to the number of different classes or categories involved in a learning task. For example, in a 5-way problem, the model is presented with instances from 5 different classes.

- K-shot: This indicates the number of examples (or "shots") from each class that the model has access to for learning. In a 1-shot learning task, the model gets only one example per class, while in a 3-shot task, it gets three examples per class.

## 2. Usage

### Datasets

This data loader can be used on several few-shot learning datasets, such as mini-ImageNet, tiered-ImageNet and Omniglot, etc. (For some datasets, they need to be refactored as follows.)

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

- Define a customized config file by users. Fill in the blank in {} in the above command.
- Available backbones: any `resnet` structures from `torchvision.models` or `kale.embed.image_cnn`.
- Example configurations can be found in `config/omniglot_resnet18_5way5shot.yaml` and `config/omniglot_resnet18_5way1shot.yaml`.
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

- `kale.loaddata.n_way_k_shot`: Dataset class for N-way-K-shot problems.
- `kale.embed.image_cnn`: Resnet feature extractors.
- `kale.pipeline.protonet`: ProtoNet trainer in pl.LightningModule class.
- `kale.predict.losses.proto_loss`: Compute the loss and accuracy for protonet.

## Reference
```
@inproceedings{snell2017prototypical,
  title={Prototypical Networks for Few-shot Learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
 }
```
