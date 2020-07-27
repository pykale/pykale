# Created by Haiping Lu from modifying https://github.com/thuml/CDAN/blob/master/pytorch/train_image.py
import os
import argparse
import warnings
import logging
import sys
from tqdm import tqdm
# No need if pykale is installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import kale.utils.da_logger as da_logger 
from copy import deepcopy
import glob
import re
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection

import kale
import torch
from torchsummary import summary

from  config_old import C
# import loaddata as du
# import optim as ou
# import kale.utils.logger as lu
import kale.utils.seed as seeding # to unify later
import kale.embed.da_feature as da_feature
import kale.predict.da_classify as da_classify
import kale.pipeline.da_systems as da_systems
import kale.loaddata.digits as digits
import kale.loaddata.multisource as multisource
# from trainer import Trainer



# Inherite and override
# class CifarIsoNet(isonet.ISONet):
def load_config():
    config_params = {
        "train_params": {
            "adapt_lambda": C.SOLVER.AD_LAMBDA,
            "adapt_lr": C.SOLVER.AD_LR,
            "lambda_init": C.SOLVER.INIT_LAMBDA,
            "nb_adapt_epochs": C.SOLVER.MAX_EPOCHS,
            "nb_init_epochs": C.SOLVER.INIT_EPOCHS,
            "init_lr": C.SOLVER.BASE_LR,
            "batch_size": C.SOLVER.BATCH_SIZE,
            "optimizer": {
                "type": C.SOLVER.TYPE,
                "optim_params": {
                    "momentum": C.SOLVER.MOMENTUM,
                    "weight_decay": C.SOLVER.WEIGHT_DECAY,
                    "nesterov": C.SOLVER.NESTEROV
                }
            }
        },
        "archi_params": {
            "feature": {
                "name": C.DATASET.NAME
            },
            "task": {
                "name": C.DATASET.NAME,
                "n_classes": C.DATASET.NUM_CLASSES
            },
            "critic": {
                "name": C.DATASET.NAME
            }    
        },
        "method_params": {
            "method": C.DAN.METHOD,            
        },
       "data_params": {
            "dataset_group": C.DATASET.NAME,
            "dataset_name": C.DATASET.SOURCE + '2' + C.DATASET.TARGET,
            "source": C.DATASET.SOURCE,
            "target": C.DATASET.TARGET,
            "size_type": C.DATASET.SIZE_TYPE,
            "weight_type": C.DATASET.WEIGHT_TYPE
        }            
    }
    return config_params

def arg_parse():
    parser = argparse.ArgumentParser(description='Domain Adversarial Networks on Digits Datasets')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--output', default='default', help='folder to save output', type=str)
    parser.add_argument('--gpus', default= '0', help='folder to save output', type=str)    
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    # ---- setup device ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Using device ' + device)

    # ---- setup configs ----
    C.merge_from_file(args.cfg)
    C.freeze()
    seed=C.SOLVER.SEED
    seeding.set_seed(seed)
    config_params = load_config()
    method_params = config_params["method_params"]
    archi_params  = config_params["archi_params"]
    train_params  = config_params["train_params"]
    data_params  = config_params["data_params"]
    archi_params["random_state"] = seed
    del method_params["method"]
    if C.DAN.METHOD is 'CDAN':
        method_params["use_random"] = C.DAN.USERANDOM 
    # ---- setup logger and output ----
    # output_dir = os.path.join(C.OUTPUT.DIR, C.DATASET.NAME + '_' + C.DATASET.SOURCE + '2' + C.DATASET.TARGET, args.output)
    output_dir = os.path.join(C.OUTPUT.DIR, C.DATASET.NAME + '_' + C.DATASET.SOURCE + '2' + C.DATASET.TARGET)
    os.makedirs(output_dir, exist_ok=True)
    
 # parameters that change across experiments for the same dataset
    record_params = train_params.copy()
    record_params.update(
        {
            k: v
            for k, v in data_params.items()
            if k not in ("dataset_group", "dataset_name")
        }
    )
    # print(type(record_params))
    # input('debug record_params')
    params_hash = da_logger.param_to_hash(record_params)
    hash_file = os.path.join(output_dir, "parameters.json")
    da_logger.record_hashes(hash_file, params_hash, record_params)
    output_file_prefix = os.path.join(output_dir, params_hash)

    test_csv_file = f"{output_file_prefix}.csv"
    checkpoint_dir = os.path.join(output_dir, "checkpoints", params_hash)

    results = da_logger.XpResults.from_file(
        ["source acc", "target acc", "domain acc"], test_csv_file
    )

    # outlogger = lu.construct_logger('dans', output_dir)
    # outlogger.info('Using ' + device)
    # outlogger.info(C.dump())    

    format_str = "@%(asctime)s %(name)s [%(levelname)s] - (%(message)s)"
    logging.basicConfig(format=format_str)
    syslogger = logging.getLogger()
    syslogger.setLevel(logging.INFO)    
    
    # ---- setup dataset ----
    source_loader, target_loader, num_channels = digits.DigitDataset.get_accesses(
        digits.DigitDataset(C.DATASET.SOURCE.upper()), digits.DigitDataset(C.DATASET.TARGET.upper()),
        C.DATASET.ROOT)
    
    
    dataset = multisource.MultiDomainDatasets(source_loader, target_loader, C.DATASET.WEIGHT_TYPE, C.DATASET.SIZE_TYPE)
    n_classes = C.DATASET.NUM_CLASSES
    data_dim = C.DATASET.DIMENSION
   
    print('==> Building model..')
    # setup feature extractor
    feature_network = da_feature.FeatureExtractorDigits(num_channels)
    # setup classifier
    feature_dim = feature_network.output_size()
    classifier_network = da_classify.DataClassifierDigits(feature_dim, n_classes)
    
    method = da_systems.Method(C.DAN.METHOD)
    train_params_local = deepcopy(train_params)
    print('==>>>>>>>>>>> Method: ' + C.DAN.METHOD)
    critic_input_size = feature_dim
    # setup critic network
    if method.is_cdan_method():
        if C.DAN.USERANDOM:
            critic_input_size = C.DAN.RANDOM_DIM
        else:
            critic_input_size = feature_dim * n_classes
    critic_network = da_classify.DomainClassifierDigits(critic_input_size)
    model = da_systems.create_dann_like(
        method=method,
        dataset=dataset,
        feature_extractor=feature_network,
        task_classifier=classifier_network,
        critic=critic_network,
        **method_params,
        **train_params_local,
    )
        
        
        # method, dataset, feature_network, classifier_network,
        # critic_network, **method_params, **train_params_local)

           
    method_name = method.value
    try_to_resume = False
    if checkpoint_dir is not None:
        path_method_name = re.sub(r"[^-/\w\.]", "_", method_name)
        full_checkpoint_dir = os.path.join(
            checkpoint_dir, path_method_name, f"seed_{seed}"
        )
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(full_checkpoint_dir, "{epoch}"),
            monitor="last_epoch",
            mode="max",
        )
        checkpoints = sorted(
            glob.glob(f"{full_checkpoint_dir}/*.ckpt"), key=os.path.getmtime
        )
        if len(checkpoints) > 0 and try_to_resume:
            last_checkpoint_file = checkpoints[-1]
        else:
            last_checkpoint_file = None
    else:
        checkpoint_callback = None
        last_checkpoint_file = None

    logger = False

    max_nb_epochs = train_params["nb_adapt_epochs"]
    # pb_refresh = 1 if len(dataset) < 1000 else 10
    pb_refresh = 5 if len(dataset) < 1000 else 50
    row_log_interval = max(10, len(dataset) // train_params_local["batch_size"] // 10)

    fast = False
    # test_params["fast"] = args.fast
    # print(logger)
    # print(checkpoint_callback)
    # print(last_checkpoint_file)
    # False
    # <pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint object at 0x7f279dfcf410>
    # None

    trainer = pl.Trainer(
        progress_bar_refresh_rate=pb_refresh,  # in steps
        row_log_interval=row_log_interval,
        min_epochs=train_params_local["nb_init_epochs"],
        max_epochs=max_nb_epochs + train_params_local["nb_init_epochs"],
        early_stop_callback=False,
        num_sanity_val_steps=5,
        check_val_every_n_epoch=1,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=last_checkpoint_file,
        gpus=args.gpus,
        logger=logger,
        weights_summary=None,  # 'full' is default
        fast_dev_run=fast,
    )

    if last_checkpoint_file is None:
        logging.info(f"Training model with {method.name} {da_logger.param_to_str(method_params)}")
    else:
        logging.info(
            f"Resuming training with {method.name} {da_logger.param_to_str(method_params)}, from {last_checkpoint_file}."
        )
    trainer.fit(model)
    if trainer.interrupted:
        raise KeyboardInterrupt("Trainer was interrupted and shutdown gracefully.")

    # if syslogger:
    #     syslogger.log_hyperparams(
    #         {"finish time": da_logger.create_timestamp_string("%Y-%m-%d %H:%M:%S")}
    #     )

    # validation scores
    results.update(
        is_validation=True,
        method_name=method_name,
        seed=seed,
        metric_values=trainer.callback_metrics,
    )
    # test scores
    trainer.test()
    results.update(
        is_validation=False,
        method_name=method_name,
        seed=seed,
        metric_values=trainer.callback_metrics,
    )
    backup_file = test_csv_file
    results.to_csv(backup_file)
    results.print_scores(
        method_name, stdout=True, fdout=None, print_func=tqdm.write,
    )
    # res_archis[seed] = trained_archi
    # progress_callback((i + 1) / nseeds)
    # print(net)
    # net = net.to(device)
 
    # if args.resume:
    #     # Load checkpoint
    #     print('==> Resuming from checkpoint..')
    #     cp = torch.load(args.resume)
    #     trainer.model.load_state_dict(cp['net'])
    #     trainer.optim.load_state_dict(cp['optim'])
    #     trainer.epochs = cp['epoch']
    #     trainer.train_acc = cp['train_accuracy']
    #     trainer.val_acc = cp['test_accuracy']

    # trainer.train()


if __name__ == '__main__':
    main()