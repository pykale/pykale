import torch
from tdc.multi_pred import DTI
from torch.utils import data

from kale.prepdata.prep_chem import integer_label_protein, integer_label_smiles


class BindingDBDataset(data.Dataset):
    def __init__(
        self,
        name: str,
        split: str = "train",
        path: str = "./data",
        mode: str = "cnn_cnn",
        y_log: bool = True,
        drug_transform=None,
        protein_transform=None
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
