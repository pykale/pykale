# Prototypical Networks for Few-shot Learning

### 1. Description

This example is constructed by modifying [Prototypical Networks for Few-shot Learning](https://github.com/jakesnell/prototypical-networks)

## 2. Usage

### Datasets 

mini-ImageNet, tiered-ImageNet, Omniglot, etc. (For some datasets, they need to be refactored as following.)

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
    |   ├── class 1
    |   |   ├── image 1
    |   |   ├── image 2
    |   |   └──...
    |   ├── class 2
    |   |   ├── image 1
    |   |   ├── image 2
    |   |   └──...
    |   └── ...
    └── test (optional)
        ├── class 1
        |   ├── image 1
        |   ├── image 2
        |   └── ...
        ├── class 2
        |   ├── image 1
        |   ├── image 2
        |   └── ...
        └── ...
</pre>

### Example Running

#### 1-gpu training:

Example - omniglot + resnet18 + 5-way-5-shot

`python main.py --cfg examples/protonet/configs/omniglot_resnet18_5way5shot.yaml --gpus 1`

Customized running

`python main.py --cfg configs/{dataset name}_{backbone}_{n}way{k}shot.yaml --gpus 1`

- Define a customized config file by users. Fill in the blank in {} in above command.
- Svailable backbones: any `resnet` structures from `torchvision.models` or `kale.embed.image_cnn`.
- An example config file is provided in `config/omniglot_resnet18_5way5shot.yaml` and `config/omniglot_resnet18_5way1shot.yaml`.
- Remember to change DATASET.ROOT item in config files to fit your dataset directory.

#### Test

Example - omniglot + resnet18 + 5-way-5-shot

`python main.py --cfg examples/protonet/configs/omniglot_resnet18_5way5shot.yaml --gpus 1 --ckpt {path to ckpt file}`

Customized running

`python main.py --cfg configs/{dataset name}_{backbone}_{n}way{k}shot.yaml --gpus 1 --ckpt {path to ckpt file}`

The test hyper-parameters are the same as config.VAL part.

If no `test` folder in custom dataset, both copying & pasting `val` subset and renaming it as `test` or changing the `mode` in defining `test_set` part in `test.py` to `val` will work.

## 3. Related `kale` API

- `kale.loaddata.n_way_k_shot`: Dataset class for N-way-K-shot problems.
- `kale.embed.image_cnn`: Resnet feature extractors.
- `kale.pipeline.protonet`: ProtoNet trainer in pl.LightningModule class.
- `kale.predict.losses.proto_loss`: Compute the loss and accuracy for protonet.





