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

    save_figs = cfg.OUTPUT.SAVE_FIG
    fig_format = cfg.SAVE_FIG_KWARGS.format
    print(f"Save Figures: {save_figs}")

    # ---- initialize folder to store images ----
    save_figures_location = cfg.OUTPUT.ROOT
    print(f"Save Figures: {save_figures_location}")

    if not os.path.exists(save_figures_location):
        os.makedirs(save_figures_location)

    # ---- setup dataset ----
    base_dir = cfg.DATASET.BASE_DIR
    file_format = cfg.DATASET.FILE_FORAMT
    download_file_by_url(cfg.DATASET.SOURCE, cfg.DATASET.ROOT, "%s.%s" % (base_dir, file_format), file_format)

    img_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.IMG_DIR)
    patient_dcm_list = read_dicom_dir(img_path, sort_instance=True, sort_patient=True)
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

    # plot the first phase of images with landmarks
    marker_names = list(landmark_df.columns[1::2])
    markers = []
    for marker in marker_names:
        marker_name = marker.split(" ")
        marker_name.pop(-1)
        marker_name = " ".join(marker_name)
        markers.append(marker_name)

    if save_figs:
        n_img_per_fig = 45
        n_figures = int(n_samples / n_img_per_fig) + 1
        for k in range(n_figures):
            visualize.plot_multi_images(
                [images[i][0, ...] for i in range(k * n_img_per_fig, min((k + 1) * n_img_per_fig, n_samples))],
                marker_locs=landmarks[k * n_img_per_fig : min((k + 1) * n_img_per_fig, n_samples), :],
                im_kwargs=dict(cfg.PLT_KWS.IM),
                marker_cmap="Set1",
                marker_kwargs=dict(cfg.PLT_KWS.MARKER),
                marker_titles=markers,
                image_titles=list(patient_ids[k * n_img_per_fig : min((k + 1) * n_img_per_fig, n_samples)]),
                n_cols=5,
            ).savefig(
                str(save_figures_location) + "/0)landmark_visualization_%s_of_%s.%s" % (k + 1, n_figures, fig_format),
                **dict(cfg.SAVE_FIG_KWARGS),
            )

    # ---- data pre-processing ----
    # ----- image registration -----
    img_reg, max_dist = reg_img_stack(images.copy(), landmarks, landmarks[0])
    plt_kawargs = {**{"im_kwargs": dict(cfg.PLT_KWS.IM), "image_titles": list(patient_ids)}, **dict(cfg.PLT_KWS.PLT)}
    if save_figs:
        visualize.plot_multi_images([img_reg[i][0, ...] for i in range(n_samples)], **plt_kawargs).savefig(
            str(save_figures_location) + "/1)image_registration.%s" % fig_format, **dict(cfg.SAVE_FIG_KWARGS)
        )

    # ----- masking -----
    img_masked = mask_img_stack(img_reg.copy(), mask)
    if save_figs:
        visualize.plot_multi_images([img_masked[i][0, ...] for i in range(n_samples)], **plt_kawargs).savefig(
            str(save_figures_location) + "/2)masking.%s" % fig_format, **dict(cfg.SAVE_FIG_KWARGS)
        )

    # ----- resize -----
    img_rescaled = rescale_img_stack(img_masked.copy(), scale=1 / cfg.PROC.SCALE)
    if save_figs:
        visualize.plot_multi_images([img_rescaled[i][0, ...] for i in range(n_samples)], **plt_kawargs).savefig(
            str(save_figures_location) + "/3)resize.%s" % fig_format, **dict(cfg.SAVE_FIG_KWARGS)
        )

    # ----- normalization -----
    img_norm = normalize_img_stack(img_rescaled.copy())
    if save_figs:
        visualize.plot_multi_images([img_norm[i][0, ...] for i in range(n_samples)], **plt_kawargs).savefig(
            str(save_figures_location) + "/4)normalize.%s" % fig_format, **dict(cfg.SAVE_FIG_KWARGS)
        )

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
    if save_figs:
        visualize.plot_weights(
            top_weights[0][0],
            background_img=images[0][0],
            im_kwargs=dict(cfg.PLT_KWS.IM),
            marker_kwargs=dict(cfg.PLT_KWS.WEIGHT),
        ).savefig(str(save_figures_location) + "/5)weights.%s" % fig_format, **dict(cfg.SAVE_FIG_KWARGS))


if __name__ == "__main__":
    main()
