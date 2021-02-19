import torch
from tdc.multi_pred import DTI
from torch.utils.data import Dataset

from kale.prepdata.chemical_transform import integer_label_protein, integer_label_smiles


class DTIDeepDataset(Dataset):
    """
    A custom dataset for loading and processing original TDC data, which is used as input data in DeepDTA model.
    Args:
         dataset (str): TDC dataset name.
         path (str): dataset download/local load path (default: "./data")
         split (str): Data split type (train, valid or test).
         transform: Transform operation (default: None)
         y_log (bool): Whether convert y values to log space. (default: True)
    """
    def __init__(self, dataset, path='./data', split="train", transform=None, y_log=True):
        self.data = DTI(name=dataset, path=path)
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
        x_drug = torch.LongTensor(integer_label_smiles(self.drug_smile[idx]))
        x_target = torch.LongTensor(integer_label_protein(self.prot_sequence[idx]))
        y = torch.Tensor([self.y[idx]])
        return x_drug, x_target, y
