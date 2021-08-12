"""
PyKale Tutorial: A Machine Learning Pipeline for PAH Diagnosis

Reference:
Swift, A. J., Lu, H., Uthoff, J., Garg, P., Cogliano, M., Taylor, J., ... & Kiely, D. G. (2020). A machine learning
cardiac magnetic resonance approach to extract disease features and automate pulmonary arterial hypertension diagnosis.
European Heart Journal-Cardiovascular Imaging. https://academic.oup.com/ehjcimaging/article/22/2/236/5717931
"""
import os

import numpy as np
import pandas as pd
from config import get_cfg_defaults
from sklearn.model_selection import cross_validate

from kale.interpret import model_weights, visualize
from kale.loaddata.get_dicom import read_dicom_images
from kale.pipeline.mpca_trainer import MPCATrainer
from kale.prepdata.image_transform import mask_img_stack, normalize_img_stack, reg_img_stack, rescale_img_stack
from kale.utils.download import download_file_by_url


def main():
    # ---- setup configs ----
    cfg_path = "configs/tutorial_svc.yaml"  # Path to `.yaml` config file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
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

    df_file = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.LANDMARK_FILE)
    df = pd.read_csv(df_file, index_col="Subject")
    landmarks = df.iloc[:, :6].values
    y = df["Group"].values
    y[np.where(y != 0)] = 1  # convert to binary classification problem, i.e. no PH vs PAH

    visualize.plot_multi_images(images[:, 0, ...], marker_locs=landmarks).show()  # plot the first phase of images

    # ---- data pre-processing ----
    # ----- image registration -----
    img_reg, max_dist = reg_img_stack(images, landmarks)
    visualize.plot_multi_images(img_reg[:, 0, ...]).show()

    # ----- masking -----
    img_masked = mask_img_stack(img_reg, mask[0, 0, ...])
    visualize.plot_multi_images(img_masked[:, 0, ...]).show()

    # ----- resize -----
    img_rescaled = rescale_img_stack(img_masked, scale=2)
    visualize.plot_multi_images(img_rescaled[:, 0, ...]).show()

    # ----- normalization -----
    img_norm = normalize_img_stack(img_rescaled)
    visualize.plot_multi_images(img_norm[:, 0, ...]).show()

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
    top_weights = model_weights.select_top_weight(weights, select_ratio=0.1)
    visualize.plot_weights(top_weights[0][0], background_img=x[0][0]).show()


if __name__ == "__main__":
    main()
