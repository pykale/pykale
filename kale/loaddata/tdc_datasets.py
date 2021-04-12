import torch
from tdc.multi_pred import DTI
from torch.utils import data

from kale.prepdata.chem_transform import integer_label_protein, integer_label_smiles


class BindingDBDataset(data.Dataset):
    """
    A custom dataset for loading and processing original TDC data, which is used as input data in DeepDTA model.

    Args:
         name (str): TDC dataset name.
         split (str): Data split type (train, valid or test).
         path (str): dataset download/local load path (default: "./data")
         mode (str): encoding mode (default: cnn_cnn)
         drug_transform: Transform operation (default: None)
         protein_transform: Transform operation (default: None)
         y_log (bool): Whether convert y values to log space. (default: True)
    """

    def __init__(
        self,
        name: str,
        split="train",
        path="./data",
        mode="cnn_cnn",
        y_log=True,
        drug_transform=None,
        protein_transform=None,
    ):
        self.data = DTI(name=name, path=path)
        self.mode = mode.lower()
        if y_log:
            self.data.convert_to_log()
        self.data = self.data.get_split()[split]
        self.drug_transform = drug_transform
        self.protein_transform = protein_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        drug, protein, label = self.data["Drug"][idx], self.data["Target"][idx], self.data["Y"][idx]
        mode_drug, mode_protein = self.mode.split("_")
        if mode_drug == "cnn":
            drug = torch.LongTensor(integer_label_smiles(drug))
        if mode_protein == "cnn":
            protein = torch.LongTensor(integer_label_protein(protein))
        label = torch.Tensor([label])
        if self.drug_transform is not None:
            self.drug_transform(drug)
        if self.protein_transform is not None:
            self.protein_transform(protein)
        return drug, protein, label
