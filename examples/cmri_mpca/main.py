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
from kale.loaddata.image_access import dicom2arraylist, read_dicom_dir
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

    save_images = cfg.OUTPUT.SAVE_IMAGES
    print(f"Save Images: {save_images}")

    # ---- initialize folder to store images ----
    save_images_location = cfg.OUTPUT.ROOT
    print(f"Save Images: {save_images_location}")

    if not os.path.exists(save_images_location):
        os.makedirs(save_images_location)

    # ---- setup dataset ----
    base_dir = cfg.DATASET.BASE_DIR
    file_format = cfg.DATASET.FILE_FORAMT
    download_file_by_url(cfg.DATASET.SOURCE, cfg.DATASET.ROOT, "%s.%s" % (base_dir, file_format), file_format)

    img_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.IMG_DIR)
    patient_dcm_list = read_dicom_dir(img_path, sort_instance=True)
    images, patient_ids = dicom2arraylist(patient_dcm_list, return_patient_id=True)
    patient_ids = np.array(patient_ids, dtype=int)
    n_samples = len(images)

    mask_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.MASK_DIR)
    mask_dcm = read_dicom_dir(mask_path, sort_instance=True)
    mask = dicom2arraylist(mask_dcm, return_patient_id=False)[0][0, ...]

    landmark_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.LANDMARK_FILE)
    landmark_df = pd.read_csv(landmark_path, index_col="Subject").loc[patient_ids]  # read .csv file as dataframe
    landmarks = landmark_df.iloc[:, :-1].values
    y = landmark_df["Group"].values
    y[np.where(y != 0)] = 1  # convert to binary classification problem, i.e. no PH vs PAH

    # plot the first phase of images
    if save_images:
        visualize.plot_multi_images(
            [images[i][0, ...] for i in range(n_samples)],
            marker_locs=landmarks,
            im_kwargs=dict(cfg.IM_KWARGS),
            marker_kwargs=dict(cfg.MARKER_KWARGS),
            n_cols=10,
        ).savefig(str(save_images_location) + "/0)first_phase.png")

    # ---- data pre-processing ----
    # ----- image registration -----
    img_reg, max_dist = reg_img_stack(images.copy(), landmarks, landmarks[0])
    if save_images:
        visualize.plot_multi_images(
            [img_reg[i][0, ...] for i in range(n_samples)], im_kwargs=dict(cfg.IM_KWARGS), n_cols=10
        ).savefig(str(save_images_location) + "/1)image_registration")

    # ----- masking -----
    img_masked = mask_img_stack(img_reg.copy(), mask)
    if save_images:
        visualize.plot_multi_images(
            [img_masked[i][0, ...] for i in range(n_samples)], im_kwargs=dict(cfg.IM_KWARGS), n_cols=10
        ).savefig(str(save_images_location) + "/2)masking")

    # ----- resize -----
    img_rescaled = rescale_img_stack(img_masked.copy(), scale=1 / cfg.PROC.SCALE)
    if save_images:
        visualize.plot_multi_images(
            [img_rescaled[i][0, ...] for i in range(n_samples)], im_kwargs=dict(cfg.IM_KWARGS), n_cols=10
        ).savefig(str(save_images_location) + "/3)resize")

    # ----- normalization -----
    img_norm = normalize_img_stack(img_rescaled.copy())
    if save_images:
        visualize.plot_multi_images(
            [img_norm[i][0, ...] for i in range(n_samples)], im_kwargs=dict(cfg.IM_KWARGS), n_cols=10
        ).savefig(str(save_images_location) + "/4)normalize")

    # ---- evaluating machine learning pipeline ----
    x = np.concatenate([img_norm[i].reshape((1,) + img_norm[i].shape) for i in range(n_samples)], axis=0)
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
    weights = mask_img_stack(weights, mask)  # masking weights
    top_weights = model_weights.select_top_weight(weights, select_ratio=0.02)  # select top 2% weights
    if save_images:
        visualize.plot_weights(
            top_weights[0][0],
            background_img=images[0][0],
            im_kwargs=dict(cfg.IM_KWARGS),
            marker_kwargs=dict(cfg.WEIGHT_KWARGS),
        ).savefig(str(save_images_location) + "/5)weights")


if __name__ == "__main__":
    main()
