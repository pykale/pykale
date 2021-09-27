"""
PAH Diagnosis from Cardiac MRI via a Multilinear PCA-based Pipeline

Reference:
Swift, A. J., Lu, H., Uthoff, J., Garg, P., Cogliano, M., Taylor, J., ... & Kiely, D. G. (2021). A machine learning
cardiac magnetic resonance approach to extract disease features and automate pulmonary arterial hypertension diagnosis.
European Heart Journal-Cardiovascular Imaging. https://academic.oup.com/ehjcimaging/article/22/2/236/5717931
"""
import argparse
import os

import numpy as np
import pandas as pd
from config import get_cfg_defaults
from sklearn.model_selection import cross_validate

from kale.interpret import model_weights, visualize
from kale.loaddata.image_access import read_dicom_images
from kale.pipeline.mpca_trainer import MPCATrainer
from kale.prepdata.image_transform import mask_img_stack, normalize_img_stack, reg_img_stack, rescale_img_stack
from kale.utils.download import download_file_by_url


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Machine learning pipeline for PAH diagnosis")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    # ---- setup dataset ----
    base_dir = cfg.DATASET.BASE_DIR
    file_format = cfg.DATASET.FILE_FORAMT
    download_file_by_url(cfg.DATASET.SOURCE, cfg.DATASET.ROOT, "%s.%s" % (base_dir, file_format), file_format)

    img_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.IMG_DIR)
    images = read_dicom_images(img_path, sort_instance=True, sort_patient=True)

    mask_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.MASK_DIR)
    mask = read_dicom_images(mask_path, sort_instance=True)

    landmark_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.LANDMARK_FILE)
    landmark_df = pd.read_csv(landmark_path, index_col="Subject")  # read .csv file as dataframe
    landmarks = landmark_df.iloc[:, :6].values
    y = landmark_df["Group"].values
    y[np.where(y != 0)] = 1  # convert to binary classification problem, i.e. no PH vs PAH

    # plot the first phase of images
    visualize.plot_multi_images(
        images[:, 0, ...], marker_locs=landmarks, im_kwargs=dict(cfg.IM_KWARGS), marker_kwargs=dict(cfg.MARKER_KWARGS)
    ).show()

    # ---- data pre-processing ----
    # ----- image registration -----
    img_reg, max_dist = reg_img_stack(images.copy(), landmarks)
    visualize.plot_multi_images(img_reg[:, 0, ...], im_kwargs=dict(cfg.IM_KWARGS)).show()

    # ----- masking -----
    img_masked = mask_img_stack(img_reg.copy(), mask[0, 0, ...])
    visualize.plot_multi_images(img_masked[:, 0, ...], im_kwargs=dict(cfg.IM_KWARGS)).show()

    # ----- resize -----
    img_rescaled = rescale_img_stack(img_masked.copy(), scale=1 / cfg.PROC.SCALE)
    visualize.plot_multi_images(img_rescaled[:, 0, ...], im_kwargs=dict(cfg.IM_KWARGS)).show()

    # ----- normalization -----
    img_norm = normalize_img_stack(img_rescaled.copy())
    visualize.plot_multi_images(img_norm[:, 0, ...], im_kwargs=dict(cfg.IM_KWARGS)).show()

    # ---- evaluating machine learning pipeline ----
    x = img_norm.copy()
    trainer = MPCATrainer(classifier=cfg.PIPELINE.CLASSIFIER, n_features=200)
    cv_results = cross_validate(trainer, x, y, cv=10, scoring=["accuracy", "roc_auc"], n_jobs=1)

    print("Averaged training time: {:.4f} seconds".format(np.mean(cv_results["fit_time"])))
    print("Averaged testing time: {:.4f} seconds".format(np.mean(cv_results["score_time"])))
    print("Averaged Accuracy: {:.4f}".format(np.mean(cv_results["test_accuracy"])))
    print("Averaged AUC: {:.4f}".format(np.mean(cv_results["test_roc_auc"])))

    # ---- model weights interpretation ----
    trainer.fit(x, y)

    weights = trainer.mpca.inverse_transform(trainer.clf.coef_) - trainer.mpca.mean_
    weights = rescale_img_stack(weights, cfg.PROC.SCALE)  # rescale weights to original shape
    weights = mask_img_stack(weights, mask[0, 0, ...])  # masking weights
    top_weights = model_weights.select_top_weight(weights, select_ratio=0.02)  # select top 2% weights
    visualize.plot_weights(
        top_weights[0][0],
        background_img=images[0][0],
        im_kwargs=dict(cfg.IM_KWARGS),
        marker_kwargs=dict(cfg.WEIGHT_KWARGS),
    ).show()


if __name__ == "__main__":
    main()
