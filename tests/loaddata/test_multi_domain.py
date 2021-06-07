import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from kale.loaddata.multi_domain import MultiDomainAdapDataset, MultiDomainImageFolder
from kale.loaddata.sampler import BalancedBatchSampler

transform = transforms.Compose(
    [
        # you can add other transformations in this list
        transforms.ToTensor()
    ]
)

# data_path = "/media/shuoz/MyDrive/data/PACS/kfold"
data_path = "D:/ML_data/PACS/kfold"

a = MultiDomainImageFolder(data_path, transform=transform, return_domain_label=True)

ds = MultiDomainAdapDataset(a, target_label=1)
ds.prepare_data_loaders()
dl = ds.get_domain_loaders(split="train")
b = Subset(a, [1, 2, 3, 4, 5])
c = ImageFolder("%s/photo" % data_path, transform=transform)
sub_sampler_a = RandomSampler(a)
sampler_a = BatchSampler(sub_sampler_a, batch_size=32, drop_last=True)
loader_a = DataLoader(dataset=a, batch_sampler=sampler_a)

sampler_c = RandomSampler(c)

sampler = BatchSampler(sampler_c, batch_size=32, drop_last=True)

loader_c = DataLoader(dataset=c, batch_sampler=sampler_c)
dl1 = DataLoader(a, batch_size=64, shuffle=False)
dl2 = DataLoader(b, batch_size=32, shuffle=False)
