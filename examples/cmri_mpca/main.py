"""
PyKale Tutorial: A Machine Learning Pipeline for PAH Diagnosis

Reference:
Swift, A. J., Lu, H., Uthoff, J., Garg, P., Cogliano, M., Taylor, J., ... & Kiely, D. G. (2020). A machine learning
cardiac magnetic resonance approach to extract disease features and automate pulmonary arterial hypertension diagnosis.
European Heart Journal-Cardiovascular Imaging. https://academic.oup.com/ehjcimaging/article/22/2/236/5717931
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import get_cfg_defaults, read_dicom_imgs
from sklearn.model_selection import cross_validate

from kale.interpret import model_weights
from kale.pipeline.mpca_trainer import MPCATrainer
from kale.prepdata.image_transform import mask_img_stack, normalize_img_stack, reg_img_stack, rescale_img_stack
from kale.utils.download import download_file_by_url


def visualize_imgs(imgs, landmarks=None):
    columns = 10
    rows = int(imgs.shape[0] / columns) + 1

    fig = plt.figure(figsize=(20, 36))

    for i in range(imgs.shape[0]):
        fig.add_subplot(rows, columns, i + 1)
        plt.axis("off")
        plt.imshow(imgs[i, 0, ...])
        if landmarks is not None:
            coords = landmarks.iloc[i, :].values.reshape((-1, 2))
            n_landmark = coords.shape[0]
            for j in range(n_landmark):
                ix = coords[j, 0]
                iy = coords[j, 1]
                plt.plot(
                    ix,
                    iy,
                    marker="o",
                    markersize=5,
                    markerfacecolor=(1, 1, 1, 0.1),
                    markeredgewidth=1.5,
                    markeredgecolor="r",
                )
        plt.title(i + 1)

    plt.show()


def main():
    # ---- setup configs ----
    cfg_path = "tutorial.yaml"  # Path to `.yaml` config file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    print(cfg)

    # ---- setup dataset ----
    base_dir = cfg.DATASET.BASE_DIR
    file_format = cfg.DATASET.FILE_FORAMT
    download_file_by_url(cfg.DATASET.SOURCE, cfg.DATASET.ROOT, "%s.%s" % (base_dir, file_format), file_format)

    img_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.IMG_DIR)
    imgs = read_dicom_imgs(img_path)

    mask_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.MASK_DIR)
    mask = read_dicom_imgs(mask_path)

    df_file = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.LANDMARK_FILE)
    df = pd.read_csv(df_file, index_col="Subject")
    landmarks = df.iloc[:, :6]
    y = df["Group"].values
    y[np.where(y != 0)] = 1

    visualize_imgs(imgs, landmarks=landmarks)

    # ---- data pre-processing ----
    # ----- image registration -----
    img_reg, max_dist = reg_img_stack(imgs, landmarks.values)
    visualize_imgs(img_reg)
    # ----- masking -----
    img_masked = mask_img_stack(img_reg, mask[0, 0, ...])
    visualize_imgs(img_masked)
    # ----- resize -----
    img_rescaled = rescale_img_stack(img_masked, scale=2)
    visualize_imgs(img_rescaled)
    # ----- normalization -----
    img_norm = normalize_img_stack(img_rescaled)
    visualize_imgs(img_norm)

    # ---- evaluating machine learning pipeline ----
    x = img_norm.copy()
    trainer = MPCATrainer(n_features=200)
    cv_results = cross_validate(trainer, x, y, cv=10, scoring=["accuracy", "roc_auc"], n_jobs=1)

    print("Accuracy: ", np.mean(cv_results["test_accuracy"]))
    print("AUC: ", np.mean(cv_results["test_roc_auc"]))

    # ---- model weights interpretation ----
    trainer.fit(x, y)

    weights = trainer.mpca.inverse_transform(trainer.clf.coef_) - trainer.mpca.mean_
    top_weights = model_weights.select_top_weight(weights, select_ratio=0.1)
    fig = model_weights.plot_weights(top_weights[0][0], background_img=x[0][0])
    plt.show()


if __name__ == "__main__":
    main()
