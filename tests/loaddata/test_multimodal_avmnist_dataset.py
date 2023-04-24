from kale.loaddata.avmnist_datasets import AVMNISTDataset
from kale.utils.download import download_file_gdrive


def test_avmnist_dataset():
    GDRIVE_ID = "1N5k-LvLwLbPBgn3GdVg6fXMBIR6pYrKb"
    ROOT = "avmnist"
    DATASET_NAME = "data.zip"
    FILE_FORMAT = "zip"
    download_file_gdrive(GDRIVE_ID, ROOT, DATASET_NAME, FILE_FORMAT)
    dataset = AVMNISTDataset(data_dir=ROOT)

    # Test train, validation, and test data loaders
    train_loader = dataset.get_train_loader()
    valid_loader = dataset.get_valid_loader()
    test_loader = dataset.get_test_loader()

    # Check if the loaders are working and have the correct dimensions
    for loader in [train_loader, valid_loader, test_loader]:
        for images, audios, labels in loader:
            assert images.shape[1:] == (1, 28, 28)
            assert audios.shape[1:] == (1, 112, 112)
            assert labels.min() >= 0
            assert labels.max() < 10
            break
