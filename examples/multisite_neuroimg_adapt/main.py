"""
Autism Detection: Domain Adaptation for Multi-Site Neuroimaging Data Analysis

Reference:
[1] Cameron Craddock, Yassine Benhajali, Carlton Chu, Francois Chouinard, Alan Evans, András Jakab, Budhachandra Singh Khundrakpam, John David Lewis, Qingyang Li, Michael Milham, Chaogan Yan, Pierre Bellec (2013). The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives. In Neuroinformatics 2013, Stockholm, Sweden.

[2] Abraham, A., Pedregosa, F., Eickenberg, M., Gervais, P., Mueller, A., Kossaifi, J., ... & Varoquaux, G. (2014). Machine learning for neuroimaging with scikit-learn. Frontiers in neuroinformatics, 14.

[3] Zhou, S., Li, W., Cox, C.R., & Lu, H. (2020). Side Information Dependence as a Regularizer for Analyzing Human Brain Conditions across Cognitive Experiments. in AAAI 2020, New York, USA. https://ojs.aaai.org//index.php/AAAI/article/view/6179

[4] Zhou, S. (2022). Interpretable Domain-Aware Learning for Neuroimage Classification (Doctoral dissertation, University of Sheffield). https://etheses.whiterose.ac.uk/31044/1/PhD_thesis_ShuoZhou_170272834.pdf
"""
import os
import numpy as np
import pandas as pd
import warnings

from config import get_cfg_defaults
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_abide_pcp
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from kale.utils.download import download_file_by_url
from kale.pipeline.multi_domain_adapter import CoIRLS
from kale.interpret import visualize

def cross_validation(x, y, covariates, estimator, domain_adaptation=False):
    results = {"Target": [], "Num_samples": [], "Accuracy": []}
    unique_covariates = np.unique(covariates)
    n_covariates = len(unique_covariates)
    enc = OneHotEncoder(handle_unknown="ignore")
    covariate_mat = enc.fit_transform(covariates.reshape(-1, 1)).toarray()

    for tgt in unique_covariates:
        idx_tgt = np.where(covariates == tgt)
        idx_src = np.where(covariates != tgt)
        x_tgt = x[idx_tgt]
        x_src = x[idx_src]
        y_tgt = y[idx_tgt]
        y_src = y[idx_src]

        if domain_adaptation:
            estimator.fit(np.concatenate((x_src, x_tgt)), y_src,
                          np.concatenate((covariate_mat[idx_src], covariate_mat[idx_tgt])))
        else:
            estimator.fit(x_src, y_src)
        y_pred = estimator.predict(x_tgt)
        results["Accuracy"].append(accuracy_score(y_tgt, y_pred))
        results["Target"].append(tgt)
        results["Num_samples"].append(x_tgt.shape[0])

    mean_acc = sum([results["Num_samples"][i] * results["Accuracy"][i] for i in range(n_covariates)])
    mean_acc /= x.shape[0]
    results["Target"].append("Average")
    results["Num_samples"].append(x.shape[0])
    results["Accuracy"].append(mean_acc)

    return pd.DataFrame(results)

def main():
    # ---- Ignore warnings ----
    warnings.filterwarnings('ignore')

    # ---- Path to `.yaml` config file ----
    cfg_path = "configs/tutorial.yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    print(cfg)

    # ---- Fetch ABIDE fMRI timeseries ----
    root_dir = cfg.DATASET.ROOT
    pipeline = cfg.DATASET.PIPELINE
    atlas = cfg.DATASET.ATLAS
    site_ids = cfg.DATASET.SITE_IDS
    abide = fetch_abide_pcp(data_dir=root_dir, pipeline=pipeline,
                            band_pass_filtering=True, global_signal_regression=False,
                            derivatives=atlas, quality_checked=False,
                            SITE_ID=site_ids,
                            verbose=1)

    # ---- Read Phenotypic data ----
    pheno_file = os.path.join(cfg.DATASET.ROOT, "ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    pheno_info = pd.read_csv(pheno_file, index_col=0)

    # ---- Read timeseries from files ----
    data_dir = os.path.join(root_dir, "ABIDE_pcp/%s/filt_noglobal" % pipeline)
    use_idx = []
    time_series = []
    for i in pheno_info.index:
        data_file_name = "%s_%s.1D" % (pheno_info.loc[i, "FILE_ID"], atlas)
        data_path = os.path.join(data_dir, data_file_name)
        if os.path.exists(data_path):
            time_series.append(np.loadtxt(data_path, skiprows=0))
            use_idx.append(i)

    # ---- Use "DX_GROUP" (autism vs control) as labels, and "SITE_ID" as covariates ----
    pheno = pheno_info.loc[use_idx, ["SITE_ID", "DX_GROUP"]].reset_index(drop=True)

    # ---- Extracting Brain Networks Features ----
    correlation_measure = ConnectivityMeasure(kind='correlation', vectorize=True)
    brain_networks = correlation_measure.fit_transform(time_series)

    # ---- Machine Learning for Multi-site Data ----
    print("Baseline")
    estimator = RidgeClassifier()
    res_df = cross_validation(brain_networks, pheno["DX_GROUP"].values, pheno["SITE_ID"].values, estimator)
    print(res_df)

    print("Domain Adaptation")
    estimator = CoIRLS(kernel=cfg.MODEL.KERNEL, lambda_=cfg.MODEL.LAMBDA_, alpha=cfg.MODEL.ALPHA)
    res_df = cross_validation(brain_networks, pheno["DX_GROUP"].values, pheno["SITE_ID"].values,
                              estimator, domain_adaptation=True)
    print(res_df)

if __name__ == "__main__":
    main()