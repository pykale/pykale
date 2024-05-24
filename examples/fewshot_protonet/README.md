# Prototypical Networks for Few-shot Learning

### 1. Description

This demo implements [Prototypical Networks for Few-shot Learning](https://github.com/jakesnell/prototypical-networks) within the `PyKale` framework.

ProtoNet is for few-shot learning problems under $N$-Way $K$-Shot settings:

**$N$-way**: The number of classes under a particular setting. The model is presented with samples from these $N$ classes and needs to classify them. For example, 3-way means the model has to classify 3 different classes.

**$K$-shot**: The number of samples for each class in the support set. For example, in a 2-shot setting, two support samples are provided per class.

**Support set**: It is a small, labeled dataset used to train the model with a few samples of each class. The support set consists of $N$ classes ($N$-way), with $K$ samples ($K$-shot) for each class. For example, under a 3-way-2-shot setting, the support set has 3 classes with 2 samples per class, totaling 6 samples.

**Query set**: It evaluates the model's ability to generalize what it has learned from the support set. It contains samples from the same $N$ classes but not included in the support set. Continuing with the 3-way-2-shot example, the query set would include additional samples from the 3 classes, which the model must classify after learning from the support set.

ProtoNet is a few-shot learning method that can be considered a clustering method. It learns a feature space where samples from the same class are close to each other and samples from different classes are far apart. The prototypes can be seen as the cluster centers, and the feature space is learned to make the samples cluster around these prototypes. But note that ProtoNet operates in a supervised learning context, where the goal is to classify data points based on labeled training examples. Clustering is typically an unsupervised learning task, where the objective is to group data points into clusters without prior knowledge of labels.


## 2. Usage

### Datasets

This model can be used on several few-shot learning datasets, such as mini-ImageNet ([official image list](https://drive.google.com/file/d/1iBu_Iqt49opXHSUNcTRU2WQas1WICLwQ/view)/[ready-to-download data on Kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet)), tiered-ImageNet ([official dataset-generating tool](https://github.com/yaoyao-liu/tiered-imagenet-tools)/[ready-to-download data on Kaggle](https://www.kaggle.com/datasets/arjun2000ashok/tieredimagenet)), and Omniglot ([official downloading code](https://github.com/brendenlake/omniglot)/[ready-to-download data on Kaggle](https://www.kaggle.com/datasets/watesoyan/omniglot)), etc. All datasets should be organized as follows.

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

Example - Training a ResNet18-based ProtoNet on Omniglot under a 5-way-5-shot setting:

`python main.py --cfg configs/demo.yaml --gpus 1`

Customized running:

`python main.py --cfg configs/template.yaml --gpus 1`

- `demo.yaml` is a demo configuration file. Change `DATASET.ROOT` to fit your dataset directory for a quick demo running.
- `template.yaml` contains all changeable hyperparameters. It allows users to customize the model accordingly.
- Available backbones: any `resnet` structures from `torchvision.models` or `kale.embed.image_cnn`.

#### Test

Example - Testing the pretrained ResNet18-based ProtoNet on unseen classes in Omniglot under a 5-way-5-shot setting:

`python eval_unseen_classes.py --cfg configs/demo.yaml --gpus 1 --ckpt {path to ckpt file}`

Customized running:

`python eval_unseen_classes.py --cfg configs/template.yaml --gpus 1 --ckpt {path to ckpt file}`

The test hyperparameters are the same as the `VAL` section of the config file.

##### Note
If no `test` folder in the dataset, choose one of the following options:
- Use the `val` set as the test set. Copy and paste the `val` folder and rename it as `test`.
- Change the `mode` in `test_set = NWayKShotDataset(..., mode="test", ...)` in `eval_unseen_classes.py` or `main.py` to `val`.

## 3. Related `kale` API

- `kale.loaddata.few_shot`: Dataset class for few-shot learning problems under $N$-way $K$-shot settings.
- `kale.embed.image_cnn`: ResNet feature extractors.
- `kale.pipeline.fewshot_trainer`: ProtoNet trainer in `pl.LightningModule` style.
- `kale.predict.losses.proto_loss`: Computing the loss and accuracy for ProtoNet.

## Reference
[Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
```
@inproceedings{snell2017prototypical,
  title={Prototypical networks for few-shot learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
 }
```
