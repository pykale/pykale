import torch
from tdc.multi_pred import DTI
from kale.utils.chemchar_label import label_isosmile, label_prot
from torch.utils.data import Dataset, DataLoader


class DTIDeepDataset(Dataset):
    def __init__(self, dataset="DAVIS", split="train", transform=None, y_log=True):
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
        xd = torch.LongTensor(label_isosmile(self.drug_smile[idx]))
        xt = torch.LongTensor(label_prot(self.prot_sequence[idx]))
        y = torch.Tensor([self.y[idx]])
        return xd, xt, y


if __name__ == "__main__":
    train_dataset = DTIDeepDataset()
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=128)
    for batch in train_loader:
        break
