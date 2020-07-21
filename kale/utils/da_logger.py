import os.path
import logging
import shutil
import pandas as pd
import numpy as np
import json
import hashlib
import glob
import re
from datetime import datetime

# import ada.utils.experimentation as xp

def param_to_str(param_dict):
    def key_val_mapper(kv):
        if isinstance(kv[1], dict):
            return param_to_str(kv[1])
        if isinstance(kv[1], float):
            return f"{kv[0]}{kv[1]:.2f}"
        if isinstance(kv[1], bool):
            return kv[0] if kv[1] else f"no-{kv[0]}"
        if isinstance(kv[1], str):
            return kv[1]
        if isinstance(kv[1], np.ndarray):
            # return "array"
            return "x".join(map(str, kv[1].flatten()))
        return f"{kv[0]}{kv[1]}"

    return "-".join(map(key_val_mapper, param_dict.items()))

def create_timestamp_string(fmt="%Y-%m-%d.%H.%M.%S.%f"):
    now = datetime.now()
    time_str = now.strftime(fmt)
    return time_str

def param_to_hash(param_dict):
    config_hash = hashlib.md5(
        json.dumps(param_dict, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return config_hash

def record_hashes(hash_file, hash_, value):
    if os.path.exists(hash_file):
        with open(hash_file, "r") as fd:
            known_hashes = json.load(fd)
    else:
        known_hashes = {}

    if hash_ not in known_hashes:
        known_hashes[hash_] = value
        with open(hash_file, "w") as fd:
            json.dump(known_hashes, fd)
        return True
    return False

class XpResults:
    @staticmethod
    def from_file(metrics, filepath):
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0)
            # time_str = xp.create_timestamp_string()
            time_str = create_timestamp_string()
            backup_file = f"{filepath}.{time_str}.bak"
            logging.info(f"Copying {filepath} to {backup_file}")
            shutil.copyfile(filepath, backup_file)
            return XpResults(metrics, df)
        else:
            return XpResults(metrics)

    def __init__(self, metrics, df=None):
        """
        Args:
            metrics (list of string): Which metrics to record.
            df (pandas.DataFrame, optional): columns are: metrics + [seed, method, split]. 
            Defaults to None.
        """
        self._metrics = metrics[:]
        if df is None:
            self._df = pd.DataFrame(columns=metrics + ["seed", "method", "split"])
        else:
            self._df = df.copy()

    def __len__(self):
        return len(self._df)

    def already_computed(self, method_name, seed):
        if len(self._df) == 0:
            return False
        ms_df = self._df.query(f"method == '{method_name}' and seed == '{seed}'")
        if len(ms_df) == 0:
            return False
        for m in self._metrics:
            if m not in ms_df.columns:
                return False
        return True

    def remove(self, method_names):
        print(method_names)
        self._df = self._df[~self._df["method"].isin(method_names)]

    def update(self, is_validation, method_name, seed, metric_values):
        split, prefix = ("Validation", "V") if is_validation else ("Test", "Te")
        results = pd.DataFrame(
            {
                k: metric_values.get(f"{prefix}_{k.replace(' ', '_')}", None)
                for k in self._metrics
            },
            index=[0],
        )
        results["seed"] = seed
        results["method"] = method_name
        results["split"] = split
        self._df = self._df.append(results, ignore_index=True)

    def get_data(self):
        return self._df

    def get_best_archi_seed(self):
        return (
            self._df.sort_values(by=self._metrics, ascending=False)
            .head(1)
            .seed.values[0]
        )

    def get_last_seed(self):
        return self._df.tail(1).seed.values[0]

    def get_mean_seed(self, mean_metric):
        if mean_metric not in self._metrics:
            raise ValueError(f"Unknown metric: {mean_metric}")
        all_res_valid = self._df.query("split=='Validation'").dropna()
        if all_res_valid.empty:
            all_res_valid = self._df.query("split=='Test'").dropna()
        all_res_valid[mean_metric] = all_res_valid[mean_metric].astype(np.float)
        tres_means = all_res_valid.groupby("method").mean()[mean_metric]
        all_seed_res = all_res_valid.pivot(
            index="seed", columns="method", values=mean_metric
        )

        def dist_to_mean(row):
            return np.mean((row - tres_means) ** 2)

        all_seed_res["dist"] = all_seed_res.apply(dist_to_mean, axis=1)
        return all_seed_res.sort_values(by="dist", ascending=True).head(1).index[0]

    def to_csv(self, filepath):
        self._df.to_csv(filepath)

    def print_scores(
        self,
        method_name,
        split="Validation",
        stdout=True,
        fdout=None,
        print_func=print,
        file_format="markdown",
    ):
        mres = self._df.query(f"method == '{method_name}' and split == '{split}'")
        nsamples = len(mres)
        mmres = [
            (mres[m].mean(), mres[m].std() / np.sqrt(nsamples)) for m in self._metrics
        ]
        if stdout:
            print_func(split, end=" ")
            print_func(method_name, end="")
            print_func(" " * (10 - len(method_name)), end="")
            print_func(
                "\t\t".join(
                    (
                        f"{m * 100:.1f}% +- {1.96 * s * 100:.2f} ({nsamples} runs)"
                        for m, s in mmres
                    )
                )
            )
        if fdout is not None:
            if file_format == "markdown":
                fdout.write(f"|{method_name}|")
                fdout.write(
                    "|".join(
                        (f"{m * 100:.1f}% +- {1.96 * s * 100:.2f}" for m, s in mmres)
                    )
                )
                fdout.write("|\n")
            else:
                fdout.write(method_name)
                fdout.write(" " * (10 - len(method_name)))
                fdout.write(
                    "\t\t".join(
                        (f"{m * 100:.1f}% +- {1.96 * s * 100:.2f}" for m, s in mmres)
                    )
                )
                fdout.write(f" ({split})")
                fdout.write("\n")

    def append_to_txt(self, filepath, test_params, nseeds, splits=None):
        if splits is None:
            splits = ["Validation", "Test"]
        with open(filepath, "a") as fd:
            fd.write(param_to_str(test_params))
            # fd.write(xp.param_to_str(test_params))
            fd.write("\n")
            print(" " * 10, "\t\t".join(self._metrics))
            fd.write("nseeds = ")
            fd.write(str(nseeds))
            fd.write("\n")
            fd.write("method\t")
            fd.write("\t\t".join(self._metrics))
            fd.write("\n")
            for name in self._df.method.unique():
                for split in splits:
                    self.print_scores(
                        method_name=name,
                        split=split,
                        stdout=True,
                        fdout=fd,
                        file_format="text",
                    )
            fd.write("\n")

    def append_to_markdown(self, filepath, test_params, nseeds, splits=None):
        if splits is None:
            splits = ["Validation", "Test"]
        with open(filepath, "a") as fd:
            fd.write(param_to_str(test_params))
            # fd.write(xp.param_to_str(test_params))
            fd.write("\n")
            print(" " * 10, "\t\t".join(self._metrics))
            fd.write("nseeds = ")
            fd.write(str(nseeds))
            fd.write("\n")
            fd.write(f"|Method|{'|'.join(self._metrics)}|\n")
            fd.write("|:----|")
            for i in range(len(self._metrics)):
                fd.write(":---:|")
            fd.write("\n")
            for name in self._df.method.unique():
                for split in splits:
                    self.print_scores(
                        method_name=name, split=split, stdout=True, fdout=fd
                    )
            fd.write("\n")
