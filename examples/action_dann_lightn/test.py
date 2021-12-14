"""This example is testing domain adaptation for action recognition, using PyTorch Lightning.
We can load and test different trained models without training.

"""

import argparse
import logging

import pytorch_lightning as pl
import torch
from config import get_cfg_defaults
from model import get_model

from kale.loaddata.video_access import VideoDataset
from kale.loaddata.video_multi_domain import VideoMultiDomainDatasets


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Domain Adversarial Networks on Action Datasets")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--gpus",
        default=1,
        help="gpu id(s) to use. None/int(0) for cpu. list[x,y] for xth, yth GPU. str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--ckpt", default="", help="pre-trained parameters for the model (ckpt files)", type=str)
    args = parser.parse_args()
    return args


def weights_update(model, checkpoint):
    """Load the pre-trained parameters to the model."""
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def main():
    """The main for this domain adaptation example, showing the workflow"""
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    # ---- setup output ----
    format_str = "@%(asctime)s %(name)s [%(levelname)s] - (%(message)s)"
    logging.basicConfig(format=format_str)
    # ---- setup dataset ----
    seed = cfg.SOLVER.SEED
    source, target, num_classes = VideoDataset.get_source_target(
        VideoDataset(cfg.DATASET.SOURCE.upper()), VideoDataset(cfg.DATASET.TARGET.upper()), seed, cfg
    )
    dataset = VideoMultiDomainDatasets(
        source,
        target,
        image_modality=cfg.DATASET.IMAGE_MODALITY,
        seed=seed,
        config_weight_type=cfg.DATASET.WEIGHT_TYPE,
        config_size_type=cfg.DATASET.SIZE_TYPE,
    )

    # ---- setup model and logger ----
    model, train_params = get_model(cfg, dataset, num_classes)
    trainer = pl.Trainer(logger=False, resume_from_checkpoint=args.ckpt, gpus=args.gpus,)

    model_test = weights_update(model=model, checkpoint=torch.load(args.ckpt))

    # test scores
    trainer.test(model=model_test)


if __name__ == "__main__":
    main()
