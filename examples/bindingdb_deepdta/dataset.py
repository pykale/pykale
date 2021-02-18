import torch
from tdc.multi_pred import DTI
from torch.utils.data import Dataset

from kale.utils.chemchar_label import label_isosmile, label_prot


class DTIDeepDataset(Dataset):
    """
    A custom dataset for loading and processing original TDC data, which is used as input data in DeepDTA model.
    Args:
         dataset (str): TDC dataset name.
         split (str): Data split type (train, valid or test).
         transform: Transform operation (default: None)
         y_log (bool): Whether convert y values to log space. (default: True)
    """
    def __init__(self, dataset, split="train", transform=None, y_log=True):
        self.data = DTI(name=dataset)
        if y_log:
            self.data.convert_to_log()
        self.data = self.data.get_split()[split]
        self.transform = transform
        self.drug_smile = self.data["Drug"]
        self.prot_sequence = self.data["Target"]
        self.y = self.data["Y"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Represent a map from index to data sample. The drug and target sequences are transformed by integer/label
        encoding.
        """
        x_drug = torch.LongTensor(label_isosmile(self.drug_smile[idx]))
        x_target = torch.LongTensor(label_prot(self.prot_sequence[idx]))
        y = torch.Tensor([self.y[idx]])
        return x_drug, x_target, y
