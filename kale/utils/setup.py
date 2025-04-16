import torch
from comet_ml import Experiment


def setup_device():
    """
    Sets up the computing device for PyTorch operations.

    Returns
    -------
    torch.device
        A CUDA device if available, otherwise CPU.

    Notes
    -----
    Also calls `torch.cuda.empty_cache()` to clear any cached GPU memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    return device


def setup_comet(**params):
    """
    Initializes and configures a Comet ML Experiment for logging.

    Parameters
    ----------
    **params : dict
        Keyword arguments for creating the `Experiment` object.
        Special keys:
        - 'log_params' : dict, optional
            Hyperparameters or settings to log to the experiment.
        - 'experiment_tag' : str, optional
            Tag to assign to the experiment.
        - 'experiment_name' : str, optional
            Name to assign to the experiment.

    Returns
    -------
    comet_ml.Experiment
        A configured Comet ML Experiment object, ready for logging.
    """
    log_params = params.pop("log_params", None)
    experiment_tag = params.pop("experiment_tag", None)
    experiment_name = params.pop("experiment_name", None)

    experiment = Experiment(
        auto_output_logging="simple",
        log_code=False,
        log_git_metadata=False,
        log_git_patch=False,
        auto_param_logging=False,
        auto_metric_logging=False,
        **params,
    )

    if log_params:
        experiment.log_parameters(log_params)

    if experiment_tag:
        experiment.add_tag(experiment_tag)

    if experiment_name:
        experiment.set_name(experiment_name)

    return experiment
