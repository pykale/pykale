"""
Author: Shuo Zhou, szhou20@sheffield.ac.uk
"""
import logging
import os
import sys

import numpy as np
from scipy.io import loadmat
from skimage import exposure, transform


def regMRI(data, reg_df, reg_id=1):
    n_sample = data.shape[-1]
    if n_sample != reg_df.shape[0]:
        logging.error("Error, registration and data not match. Please check")
        sys.exit()
    n_landmark = int((reg_df.shape[1] - 1) / 2)
    # reg_target = data[..., 0, reg_id]
    _dst = reg_df.iloc[reg_id, 2:].values
    _dst = _dst.reshape((n_landmark, 2))
    max_dist = np.zeros(n_sample)
    for i in range(n_sample):
        if i == reg_id:
            continue
        else:
            # epts = reg_df.iloc[i, 1:].values
            _src = reg_df.iloc[i, 2:].values
            _src = _src.reshape((n_landmark, 2))
            idx_valid = np.isnan(_src[:, 0])
            tform = transform.estimate_transform(ttype="similarity", src=_src[~idx_valid, :], dst=_dst[~idx_valid, :])
            # forward transform used here, inverse transform used for warp
            src_tform = tform(_src[~idx_valid, :])
            dist = np.linalg.norm(src_tform - _dst[~idx_valid, :], axis=1)
            max_dist[i] = dist.max()

            for j in range(data[..., i].shape[-1]):
                src_img = data[..., j, i].copy()
                warped = transform.warp(src_img, inverse_map=tform.inverse, preserve_range=True)
                data[..., j, i] = warped

    return data, max_dist


def rescale_cmr(data, scale=16):
    n_sub = data.shape[-1]
    n_time = data.shape[-2]
    scale_ = 1 / scale
    data_all = []
    for i in range(n_sub):
        data_sub = []
        for j in range(n_time):
            img = data[:, :, j, i]
            img_ = transform.rescale(img, scale_, preserve_range=True)
            # preserve_range should be true otherwise the output will be normalised values
            data_sub.append(img_.reshape(img_.shape + (1,)))
        data_sub = np.concatenate(data_sub, axis=-1)
        data_all.append(data_sub.reshape(data_sub.shape + (1,)))
    data_all = np.concatenate(data_all, axis=-1)

    return data_all


def mat2gray(A):
    amin = np.amin(A)
    amax = np.amax(A)
    diff = amax - amin
    return (A - amin) / diff


def preproc(data, mask, level=1):
    n_sub = data.shape[-1]
    n_time = data.shape[-2]
    data_all = []
    for i in range(n_sub):
        data_sub = []
        for j in range(n_time):
            img = data[:, :, j, i]
            img_ = mat2gray(np.multiply(img, mask))
            if level == 2:
                img_ = exposure.equalize_adapthist(img_)
            data_sub.append(img_.reshape(img_.shape + (1,)))
        data_sub = np.concatenate(data_sub, axis=-1)
        data_all.append(data_sub.reshape(data_sub.shape + (1,)))
    data_all = np.concatenate(data_all, axis=-1)

    return data_all


def scale_cmr_mask(mask, scale):
    mask = transform.rescale(mask.astype("bool_"), 1 / scale, anti_aliasing=False)
    # change mask dtype to bool to ensure the output values are 0 and 1
    # anti_aliasing False otherwise the output might be all 0s
    # the following three lines should be equivalent
    # size0 = int(mask.shape[0] / scale)
    # size1 = int(mask.shape[1] / scale)
    # mask = transform.downscale_local_mean(mask, (size0, size1))
    mask_new = np.zeros(mask.shape)
    mask_new[np.where(mask > 0.5)] = 1
    return mask_new


def cmr_proc(basedir, db, scale, mask_id, level, save_data=True, return_data=False):
    logging.info("Preprocssing Data Scale: 1/%s, Mask ID: %s, Processing level: %s" % (scale, mask_id, level))
    datadir = os.path.join(basedir, "DB%s/NoPrs%sDB%s.npy" % (db, scale, db))
    maskdir = os.path.join(basedir, "Prep/AllMasks.mat")
    data = np.load(datadir)
    masks = loadmat(maskdir)["masks"]
    mask = masks[mask_id - 1, db - 1]
    mask = scale_cmr_mask(mask, scale)
    data_proc = preproc(data, mask, level)

    if save_data:
        out_path = os.path.join(basedir, "DB%s/PrepData/PrS%sM%sL%sDB%s.npy" % (db, scale, mask_id, level, db))
        np.save(out_path, data_proc)
    if return_data:
        return data_proc
