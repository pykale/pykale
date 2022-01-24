"""Logging functions, based on https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/logger.py"""

import datetime
import json
import logging
import os
import uuid


def out_file_core():
    """Creates an output file name concatenating a formatted date and uuid, but without an extension.

    Returns:
        string: A string to be used in a file name.
    """
    date = str(datetime.datetime.now().strftime("%Y%d%m_%H%M%S"))
    return f"log-{date}-{str(uuid.uuid4())}"


def construct_logger(name, save_dir):
    """Constructs a logger. Saves the output as a text file at a specified path. Also saves the output of `git diff HEAD` to the same folder.

    Reference: https://docs.python.org/3/library/logging.html

    Args:
        name (str): the logger name, typically the method name
        save_dir (str): the path to save the log file (.txt)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_no_ext = out_file_core()

    fh = logging.FileHandler(os.path.join(save_dir, file_no_ext + ".txt"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    gitdiff_patch = os.path.join(save_dir, file_no_ext + ".gitdiff.patch")
    os.system(f"git diff HEAD > {gitdiff_patch}")

    return logger


def create_multi_results_dict(pred_verb_cpu, pred_noun_cpu, pred_id, name1="verb", name2="noun"):
    """Construct a dictionary to save multi class results (verb and noun).
    Args:
        pred_verb_cpu (list): results for verb class.
        pred_noun_cpu (list): results for noun class.
        pred_id (tuple): corresponding segment ids to results.
        name1 (string): the name of the first class.
        name2 (string): the name of the second class.
    Returns:
        results_dict (dict): the dictionary covering class names and results.
    """
    results_dict = {}
    for p_verb, p_noun, p_id in zip(pred_verb_cpu, pred_noun_cpu, pred_id):
        verb_dict = {}
        noun_dict = {}
        for i, prob in enumerate(p_verb):
            verb_dict[str(i)] = prob
        for i, prob in enumerate(p_noun):
            noun_dict[str(i)] = prob
        results_dict[p_id] = {name1: verb_dict, name2: noun_dict}
    return results_dict


def create_single_result_dict(pred_cpu, pred_id, name="verb"):
    """Construct a dictionary to save single class results.
    Args:
        pred_cpu (list): results for the class.
        pred_id (tuple): corresponding segment ids to results.
        name (string): the name of the class.
    Returns:
        results_dict (dict): the dictionary covering class name and results.
    """
    results_dict = {}
    for p, p_id in zip(pred_cpu, pred_id):
        d = {}
        for i, prob in enumerate(p):
            d[str(i)] = prob
        results_dict[p_id] = {name: d}
    return results_dict


def save_results_to_json(
        y_hat, y_t_hat, y_ids, y_t_ids, y_hat_noun=None, y_t_hat_noun=None, verb=True, noun=False,
        file_name="test.json",
):
    """Save the output for each class to a json file.
    Args:
        y_hat (list): results for verb classes from the source data.
        y_t_hat (list): results for verb classes from the target data.
        y_ids (list): corresponding segment ids to source data results.
        y_t_ids (list): corresponding segment ids to target data results.
        y_hat_noun (list): results for noun classes from the source data.
        y_t_hat_noun (list): results for noun classes from the target data.
        verb (bool): check if results covers verb class.
        noun (bool): check if results covers noun class.
        file_name (string): the name of the file to save.
    """
    if verb:
        pred_verb_cpu = y_hat
        pred_t_verb_cpu = y_t_hat
    if noun:
        pred_noun_cpu = y_hat_noun
        pred_t_noun_cpu = y_t_hat_noun

    if verb and noun:
        results_dict = create_multi_results_dict(pred_verb_cpu, pred_noun_cpu, y_ids)
        results_t_dict = create_multi_results_dict(pred_t_verb_cpu, pred_t_noun_cpu, y_t_ids)
    elif verb and not noun:
        results_dict = create_single_result_dict(pred_verb_cpu, y_ids)
        results_t_dict = create_single_result_dict(pred_t_verb_cpu, y_t_ids)

    with open(file_name, "w") as f:
        json.dump(
            {
                "results_source": results_dict,
                "results_target": results_t_dict,
                "version": "0.2",
                "challenge": "domain_adaptation",
                "sls_pt": 2,
                "sls_tl": 3,
                "sls_td": 3,
            },
            f,
        )
