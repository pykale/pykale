import pytest
import torch

from kale.loaddata.multi_domain import MultiDomainDataFolder

data_path = "/media/shuoz/MyDrive/data/PACS/raw"

MultiDomainDataFolder(data_path)
