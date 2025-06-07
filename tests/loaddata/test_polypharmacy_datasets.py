from kale.loaddata.polypharmacy_datasets import PolypharmacyDataset


def test_polypharmacy_datasets():
    url = "https://github.com/pykale/data/raw/main/graphs/pose_pyg_2.pt"
    root = "./tests/test_data/polypharmacy/"
    dataset = PolypharmacyDataset(url=url, root=root, name="pose", mode="train")
    dataset.load_data()
    _ = dataset.__getitem__(0)
    assert dataset.__len__() == 1
