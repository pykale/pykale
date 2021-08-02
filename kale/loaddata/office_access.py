import logging
import os

from kale.loaddata.dataset_access import DatasetAccess
from kale.loaddata.multi_domain import MultiDomainImageFolder
from kale.prepdata.image_transform import get_transform
from kale.utils.download import download_file_by_url

url = "https://github.com/sz144/data/raw/main/image_data/office/"
DOMAINS = ["amazon", "caltech", "dslr", "webcam"]
office_transform = get_transform("office")


class OfficeAccess(MultiDomainImageFolder, DatasetAccess):
    def __init__(self, root, transform=office_transform, download=False, **kwargs):
        """Init office dataset."""
        # init params
        if download:
            self.download(root)
        super(OfficeAccess, self).__init__(root, transform=transform, **kwargs)

    @staticmethod
    def download(path):
        """Download dataset."""
        if not os.path.exists(path):
            os.makedirs(path)
        for domian_ in DOMAINS:
            filename = "%s.zip" % domian_
            data_path = os.path.join(path, filename)
            if os.path.exists(data_path):
                logging.info(f"Data file {filename} already exists.")
                continue
            else:
                data_url = "%s/%s" % (url, filename)
                download_file_by_url(data_url, path, filename, "zip")
                # zip_file = zipfile.ZipFile(data_path, "r")
                # zip_file.extractall(path)
                logging.info(f"Download {data_url} to {data_path}")

        logging.info("[DONE]")
        return


class Office31(OfficeAccess):
    def __init__(self, root, transform=office_transform, download=False, **kwargs):
        sub_domain_set = ["amazon", "dslr", "webcam"]
        super(Office31, self).__init__(
            root, transform=transform, download=download, sub_domain_set=sub_domain_set, **kwargs
        )


class OfficeCaltech(OfficeAccess):
    def __init__(self, root, transform=office_transform, download=False, **kwargs):
        sub_class_set = [
            "mouse",
            "calculator",
            "back_pack",
            "keyboard",
            "monitor",
            "projector",
            "headphones",
            "bike",
            "laptop_computer",
            "mug",
        ]
        super(OfficeCaltech, self).__init__(
            root, transform=transform, download=download, sub_class_set=sub_class_set, **kwargs
        )
