import logging
import os

from kale.loaddata.dataset_access import DatasetAccess
from kale.loaddata.multi_domain import MultiDomainImageFolder
from kale.prepdata.image_transform import get_transform
from kale.utils.download import download_file_by_url

DOMAINS = ["amazon", "caltech", "dslr", "webcam"]
office_transform = get_transform("office")


class OfficeAccess(MultiDomainImageFolder, DatasetAccess):
    """Common API for office dataset access

    Args:
        root (string): root directory of dataset
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. Defaults to office_transform.
        download (bool, optional): Whether to allow downloading the data if not found on disk. Defaults to False.

    References:
        [1] Saenko, K., Kulis, B., Fritz, M. and Darrell, T., 2010, September. Adapting visual category models to
        new domains. In European Conference on Computer Vision (pp. 213-226). Springer, Berlin, Heidelberg.
        [2] Griffin, Gregory and Holub, Alex and Perona, Pietro, 2007. Caltech-256 Object Category Dataset.
        California Institute of Technology. (Unpublished).
        https://resolver.caltech.edu/CaltechAUTHORS:CNS-TR-2007-001.
        [3] Gong, B., Shi, Y., Sha, F. and Grauman, K., 2012, June. Geodesic flow kernel for unsupervised
        domain adaptation. In IEEE Conference on Computer Vision and Pattern Recognition (pp. 2066-2073).
    """

    def __init__(self, root, transform=office_transform, download=False, **kwargs):
        if download:
            self.download(root)
        super(OfficeAccess, self).__init__(root, transform=transform, **kwargs)

    @staticmethod
    def download(path):
        """Download dataset.
            Office-31 source: https://www.cc.gatech.edu/~judy/domainadapt/#datasets_code
            Caltech-256 source: http://www.vision.caltech.edu/Image_Datasets/Caltech256/
            Data with this library is adapted from: http://www.stat.ucla.edu/~jxie/iFRAME/code/imageClassification.rar
        """
        url = "https://github.com/pykale/data/raw/main/images/office/"

        if not os.path.exists(path):
            os.makedirs(path)
        for domain_ in DOMAINS:
            filename = "%s.zip" % domain_
            data_path = os.path.join(path, filename)
            if os.path.exists(data_path):
                logging.info(f"Data file {filename} already exists.")
                continue
            else:
                data_url = "%s/%s" % (url, filename)
                download_file_by_url(data_url, path, filename, "zip")
                logging.info(f"Download {data_url} to {data_path}")

        logging.info("[DONE]")
        return


class Office31(OfficeAccess):
    def __init__(self, root, **kwargs):
        """Office-31 Dataset. Consists of three domains: 'amazon', 'dslr', and 'webcam', with 31 image classes.

        Args:
            root (string): path to directory where the office folder will be created (or exists).

        Reference:
            Saenko, K., Kulis, B., Fritz, M. and Darrell, T., 2010, September. Adapting visual category models to new
            domains. In European Conference on Computer Vision (pp. 213-226). Springer, Berlin, Heidelberg.
        """
        sub_domain_set = ["amazon", "dslr", "webcam"]
        super(Office31, self).__init__(root, sub_domain_set=sub_domain_set, **kwargs)


class OfficeCaltech(OfficeAccess):
    def __init__(self, root, **kwargs):
        """Office-Caltech-10 Dataset. This dataset consists of four domains: 'amazon', 'caltech', 'dslr', and 'webcam',
            which are samples with overlapped 10 classes between Office-31 and Caltech-256.

        Args:
            root (string): path to directory where the office folder will be created (or exists).

        References:
            [1] Saenko, K., Kulis, B., Fritz, M. and Darrell, T., 2010, September. Adapting visual category models to
            new domains. In European Conference on Computer Vision (pp. 213-226). Springer, Berlin, Heidelberg.
            [2] Griffin, Gregory and Holub, Alex and Perona, Pietro, 2007. Caltech-256 Object Category Dataset.
            California Institute of Technology. (Unpublished).
            https://resolver.caltech.edu/CaltechAUTHORS:CNS-TR-2007-001.
            [3] Gong, B., Shi, Y., Sha, F. and Grauman, K., 2012, June. Geodesic flow kernel for unsupervised
            domain adaptation. In IEEE Conference on Computer Vision and Pattern Recognition (pp. 2066-2073).
        """
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
        super(OfficeCaltech, self).__init__(root, sub_class_set=sub_class_set, **kwargs)
