# import os
# import sys
# import kale
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from kale.embed import image_cnn, mpca, positional_encoding, attention_cnn
# from kale.loaddata import (
#     cifar_access,
#     dataset_access,
#     digits_access,
#     mnistm,
#     multi_domain,
#     sampler,
#     usps,
#     videos,
# )

# from kale.pipeline import domain_adapter # need pytorch-lightning
# from kale.predict import class_domain_nets, isonet, losses
# from kale.prepdata import image_transform, prep_cmr, tensor_reshape, video_transform
# from kale.utils import csv_logger, logger, seed
from kale.utils import seed

seed.set_seed(2020)

""" These only work with the optional graph modules
from kale.embed import gcn, gripnet
"""
print("kale imported")
