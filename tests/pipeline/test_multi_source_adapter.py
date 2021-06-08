import pytest
# import torch
# from torch.utils.data import DataLoader, Subset
# from torch.utils.data.sampler import BatchSampler, RandomSampler
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from kale.embed.image_cnn import ResNet18Feature
from kale.loaddata.multi_domain import MultiDomainAdapDataset, MultiDomainImageFolder
from kale.prepdata.image_transform import get_transform
from kale.pipeline.multi_source_adapter import M3SDATrainer
from kale.predict.class_domain_nets import ClassNetSmallImage

# transform = transforms.Compose(
#     [
#         # you can add other transformations in this list
#         transforms.ToTensor()
#     ]
# )


@pytest.fixture(scope="module")
def testing_cfg(download_path):
    config_params = {
        "train_params": {
            "adapt_lambda": True,
            "adapt_lr": True,
            "lambda_init": 1,
            "nb_adapt_epochs": 2,
            "nb_init_epochs": 1,
            "init_lr": 0.001,
            "batch_size": 100,
            "optimizer": {"type": "SGD", "optim_params": {"momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}},
        }
    }
    yield config_params


DATA_PATH = ["/media/shuoz/MyDrive/data/office/office_caltech_10"]
# DATA_PATH = ["D:\ML_data\office\office_caltech_10"]
NUM_CLASSES = 10


@pytest.mark.parametrize("data_path", DATA_PATH)
def test_multi_source(data_path, testing_cfg):
    transform = get_transform("office")
    data_access = MultiDomainImageFolder(data_path, transform=transform, return_domain_label=True)
    dataset = MultiDomainAdapDataset(data_access)
    # num_channels = 3
    feature_network = ResNet18Feature()
    # setup classifier
    feature_dim = feature_network.output_size()
    classifier_network = ClassNetSmallImage(feature_dim, NUM_CLASSES)
    train_params = testing_cfg["train_params"]
    model = M3SDATrainer(dataset=dataset, feature_extractor=feature_network, task_classifier=classifier_network,
                         target_label=1, k_moment=3, **train_params)
    trainer = pl.Trainer(min_epochs=train_params["nb_init_epochs"], max_epochs=train_params["nb_adapt_epochs"],
                         gpus=None)
    trainer.fit(model)

# data_path = "/media/shuoz/MyDrive/data/PACS/kfold"
# data_path = "D:/ML_data/PACS/kfold"
# data_path = "/media/shuoz/MyDrive/data/office/office_caltech_10"
# a = MultiDomainImageFolder(data_path, transform=transform, return_domain_label=True)
#
# ds = MultiDomainAdapDataset(a)
# ds.prepare_data_loaders()
# dl = ds.get_domain_loaders(split="train")
# b = Subset(a, [1, 2, 3, 4, 5])
# c = ImageFolder("%s/photo" % data_path, transform=transform)
# sub_sampler_a = RandomSampler(a)
# sampler_a = BatchSampler(sub_sampler_a, batch_size=32, drop_last=True)
# loader_a = DataLoader(dataset=a, batch_sampler=sampler_a)
#
# sampler_c = RandomSampler(c)
#
# sampler = BatchSampler(sampler_c, batch_size=32, drop_last=True)
#
# loader_c = DataLoader(dataset=c, batch_sampler=sampler_c)
# dl1 = DataLoader(a, batch_size=64, shuffle=False)
# dl2 = DataLoader(b, batch_size=32, shuffle=False)
