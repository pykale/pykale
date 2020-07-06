# Created by Haiping Lu directly from https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/logger.py
# Under the MIT License
import os
import sys
import shlex
import logging
import datetime
import subprocess

def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def construct_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    date = str(datetime.datetime.now().strftime('%m%d%H'))
    fh = logging.FileHandler(os.path.join(save_dir, f'log-{date}-{git_hash()}.txt'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    os.system(f"git diff HEAD > {save_dir}/gitdiff.patch")

    return logger
