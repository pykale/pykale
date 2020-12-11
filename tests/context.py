import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import kale
from kale.embed import image_cnn, mpca, positional_encoding, attention_cnn
from kale.loaddata import cifar_access, dataset_access, digits_access, mnistm, multi_domain, sampler, usps, videos
from kale.pipeline import domain_adapter
from kale.predict import class_domain_nets, isonet, losses
from kale.prepdata import image_transform, prep_cmr, tensor_reshape, video_transform
from kale.utils import csv_logger, logger, seed

""" These only work with the optional graph modules
from kale.embed import gcn, gripnet
"""
print('kale imported')