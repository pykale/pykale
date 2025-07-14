import os
import sys

import pandas as pd
from torch.utils.data import DataLoader

from kale.embed.ban import DrugBAN
from kale.loaddata.molecular_datasets import DTIDataset
from kale.pipeline.drugban_trainer import DrugbanTrainer

sys.path.append("../../../pykale/")
from kale.loaddata.sampler import MultiDataLoader


def get_dataset(dataFolder, da_task, **kwargs):
    if not da_task:
        df_train = pd.read_csv(os.path.join(dataFolder, "train.csv"))
        df_val = pd.read_csv(os.path.join(dataFolder, "val.csv"))
        df_test = pd.read_csv(os.path.join(dataFolder, "test.csv"))

        train_dataset = DTIDataset(df_train.index.values, df_train)
        valid_dataset = DTIDataset(df_val.index.values, df_val)
        test_dataset = DTIDataset(df_test.index.values, df_test)

        return train_dataset, valid_dataset, test_dataset

    else:
        df_train_source = pd.read_csv(os.path.join(dataFolder, "source_train.csv"))
        df_train_target = pd.read_csv(os.path.join(dataFolder, "target_train.csv"))
        df_test_target = pd.read_csv(os.path.join(dataFolder, "target_test.csv"))

        train_source_dataset = DTIDataset(df_train_source.index.values, df_train_source)
        train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target)
        test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)

        return train_source_dataset, train_target_dataset, test_target_dataset


def get_dataloader(*datasets, batchsize, num_workers, collate_fn, is_da, da_task, **kwargs):
    params = {
        "batch_size": batchsize,
        "shuffle": True,
        "num_workers": num_workers,
        "drop_last": True,
        "collate_fn": collate_fn,
    }

    if not is_da:
        # If domain adaptation is not used
        if not da_task:
            train_dataset, valid_dataset, test_dataset = datasets
            # If in-domain splitting strategy is used
            train_dataloader = DataLoader(train_dataset, **params)
            params.update({"shuffle": False, "drop_last": False})
            valid_dataloader = DataLoader(valid_dataset, **params)
            test_dataloader = DataLoader(test_dataset, **params)

        else:
            train_dataset, _, test_target_dataset = datasets
            # If cross-domain splitting strategy is used
            train_dataloader = DataLoader(train_dataset, **params)
            params.update({"shuffle": False, "drop_last": False})
            valid_dataloader = DataLoader(test_target_dataset, **params)
            test_dataloader = DataLoader(test_target_dataset, **params)
    else:
        # If domain adaptation is used, and cross-domain splitting strategy is used
        train_dataset, train_target_dataset, test_target_dataset = datasets
        source_dataloader = DataLoader(train_dataset, **params)
        target_dataloader = DataLoader(train_target_dataset, **params)
        n_batches = max(len(source_dataloader), len(target_dataloader))
        train_dataloader = MultiDataLoader(
            dataloaders=[source_dataloader, target_dataloader], n_batches=n_batches
        )  # used to be named as multi_generator

        params.update({"shuffle": False, "drop_last": False})
        valid_dataloader = DataLoader(
            test_target_dataset, **params
        )  # validation set is the same as test set, as in the paper
        test_dataloader = DataLoader(test_target_dataset, **params)

    return train_dataloader, valid_dataloader, test_dataloader


def get_model(config, **kwargs):
    return DrugbanTrainer(
        model=DrugBAN(**config),
        solver_lr=config.SOLVER.LEARNING_RATE,
        num_classes=config.DECODER.BINARY,
        batch_size=config.SOLVER.BATCH_SIZE,
        # --- domain adaptation parameters ---
        is_da=config.DA.USE,
        solver_da_lr=config.SOLVER.DA_LEARNING_RATE,
        da_init_epoch=config.DA.INIT_EPOCH,
        da_method=config.DA.METHOD,
        original_random=config.DA.ORIGINAL_RANDOM,
        use_da_entropy=config.DA.USE_ENTROPY,
        da_random_layer=config.DA.RANDOM_LAYER,
        # --- discriminator parameters ---
        da_random_dim=config.DA.RANDOM_DIM,
        decoder_in_dim=config.DECODER.IN_DIM,
    )


def get_model_from_ckpt(ckpt_path, config):
    return DrugbanTrainer.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model=DrugBAN(**config),
        solver_lr=config.SOLVER.LEARNING_RATE,
        num_classes=config.DECODER.BINARY,
        batch_size=config.SOLVER.BATCH_SIZE,
        # --- domain adaptation parameters ---
        is_da=config.DA.USE,
        solver_da_lr=config.SOLVER.DA_LEARNING_RATE,
        da_init_epoch=config.DA.INIT_EPOCH,
        da_method=config.DA.METHOD,
        original_random=config.DA.ORIGINAL_RANDOM,
        use_da_entropy=config.DA.USE_ENTROPY,
        da_random_layer=config.DA.RANDOM_LAYER,
        # --- discriminator parameters ---
        da_random_dim=config.DA.RANDOM_DIM,
        decoder_in_dim=config.DECODER.IN_DIM,
    )


def get_test_dataset(dataFolder):
    df_test_target = pd.read_csv(dataFolder)
    test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)
    return test_target_dataset


def get_test_dataloader(dataset, batchsize, num_workers, collate_fn):
    test_dataloader = DataLoader(
        dataset, batch_size=batchsize, num_workers=num_workers, collate_fn=collate_fn, shuffle=False, drop_last=True
    )
    return test_dataloader
