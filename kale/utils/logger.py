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


def save_result_to_json(pred_verb, pred_noun, val_id, file_name="test.json"):
    pred_verb_cpu = pred_verb.cpu().tolist()
    pred_noun_cpu = pred_noun.cpu().tolist()

    results_dict = {}
    for p_verb, p_noun, id in zip(pred_verb_cpu, pred_noun_cpu, val_id):
        verb_dict = {}
        noun_dict = {}
        for i, prob in enumerate(p_verb):
            verb_dict[str(i)] = prob
        for i, prob in enumerate(p_noun):
            noun_dict[str(i)] = prob
        results_dict[id] = {"verb": verb_dict, "noun": noun_dict}

    with open(file_name, "w") as f:
        json.dump(
            {
                "results_target": results_dict,
                "version": "0.2",
                "challenge": "domain_adaptation",
                "sls_pt": 0,
                "sls_tl": 0,
                "sls_td": 0,
            },
            f,
        )
