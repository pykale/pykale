import torch
from comet_ml import Experiment


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    return device


def setup_comet(**params):
    log_params = params.pop("log_params", None)
    experiment_tag = params.pop("experiment_tag", None)
    experiment_name = params.pop("experiment_name", None)

    experiment = Experiment(**params)

    if log_params:
        experiment.log_parameters(log_params)

    if experiment_tag:
        experiment.add_tag(experiment_tag)

    if experiment_name:
        experiment.set_name(experiment_name)

    return experiment
