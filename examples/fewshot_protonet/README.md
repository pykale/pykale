# Prototypical Networks for Few-shot Learning

### 1. Description

This demo implements [Prototypical Networks for Few-shot Learning](https://github.com/jakesnell/prototypical-networks) within the `PyKale` framework.

ProtoNet is for few-shot learning problems under $N$-Way $K$-Shot settings:

**$N$-way**: The number of classes involved in a particular few-shot learning problem. It is only functional in meta-testing stage. Essentially, it defines the breadth of the classification task. For example. 5-way means the model has to distinguish between 5 different classes. In the context of few-shot learning, the model is presented with examples from these N classes and needs to learn to differentiate between them.

**$K$-shot**: The number of samples (referred to as "shots") from each class in support set. It should be the same in meta-training and meta-testing. It defines the depth of the learning task, i.e., how many instances the model has for learning each class. A 1-shot learning task indicates only one support sample per class, while a 3-shot task has three support samples per class.

**Support set**: It is a small, labeled dataset used to train the model on a few examples of each class. In meta-testing, the support set consists of N classes (N-way), with K examples (K-shot) for each class. For example, in a 3-way 2-shot task, the support set would include 3 classes with 2 examples per class, totaling 6 examples.

**Query set**: It is used to evaluate the model's ability to generalize what it has learned from the support set. It contains examples from the same N classes but does not include the examples from the support set. Continuing with the 3-way 2-shot example, the query set would include additional examples from the 3 classes, which the model must classify after learning from the support set.


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
