"""
Author: Shuo Zhou, szhou20@sheffield.ac.uk

Functions used for CMR image preprocessing:
    1. regMRI: image registration
    2. rescalse_cmr: rescale cmr image
    3. preproc: preprocessing image data
"""
import os
import sys
import numpy as np
from scipy.io import loadmat
from skimage import exposure, transform


def regMRI(data, reg_df, reg_id=1):
    """Align CMR images towards a target sample

    Args:
        data (ndarray): input data, shape (dim1, dim2, n_slice, n_subject)
        reg_df (dataframe): landmark locations for registration, shape (n_subject, 2 + n_landmark * 2)
        reg_id (int, optional): index of subject used as target sample. Defaults to 1.

    Returns:
        ndarray: Processed data, shape (dim1, dim2, n_slice, n_subject)
    """
    n_sample = data.shape[-1]
    if n_sample != reg_df.shape[0]:
        print('Error, registration loactions and image data not match. Please check')
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
            tform = transform.estimate_transform(ttype='similarity',
                                                 src=_src[~idx_valid, :],
                                                 dst=_dst[~idx_valid, :])
            # forward transform used here, inverse transform used for warp
            src_tform = tform(_src[~idx_valid, :])
            dist = np.linalg.norm(src_tform - _dst[~idx_valid, :], axis=1)
            max_dist[i] = dist.max()

            for j in range(data[..., i].shape[-1]):
                src_img = data[..., j, i].copy()
                warped = transform.warp(src_img, inverse_map=tform.inverse,
                                        preserve_range=True)
                data[..., j, i] = warped

    return data, max_dist


def rescale_cmr(data, scale=16):
    """Rescaling CMR data

    Args:
        data (ndarray): CMR data, shape (dim1, dim2, n_slice, n_subject)
        scale (int, optional): Scale factor. Defaults to 16.

    Returns:
        ndarray: rescaled CMR images, shape (dim1 / scale, dim2 / scale, n_slice, n_subject)
    """
    n_sub = data.shape[-1]
    n_time = data.shape[-2]
    scale_ = 1/scale
    data_all = []
    for i in range(n_sub):
        data_sub = []
        for j in range(n_time):
            img = data[:, :, j, i]
            img_ = transform.rescale(img, scale_, preserve_range=True)
            # preserve_range should be true otherwise the output will be normalised values
            data_sub.append(img_.reshape(img_.shape+(1,)))
        data_sub = np.concatenate(data_sub, axis=-1)
        data_all.append(data_sub.reshape(data_sub.shape+(1,)))
    data_all = np.concatenate(data_all, axis=-1)

    return data_all


def mat2gray(A):
    """Convert an image to gray scale

    Args:
        A (ndarray): input image, shape (dim1, dim2)

    Returns:
        ndarray: image where all values are between 0 and 1
    """
    amin = np.amin(A)
    amax = np.amax(A)
    diff = amax - amin
    return (A - amin) / diff


def preproc(data, mask, level=1):
    """Preprocessing CMR image data

    Args:
        data (ndarray): iuput image data, shape (dim1, dim2, n_slice, n_subject)
        mask (ndarray): mask, shape (dim1, dim2)
        level (int, optional): preprocessing level, by default 1. Defaults to 1.

    Returns:
        ndarray: preprocessed data, shape (dim1, dim2, n_slice, n_subject)
    """
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
    """Rescale mask

    Args:
        mask (ndarray): mask matrix, shape (dim1, dim2)
        scale (int): scale factors, e.g. the output will be 1/4 of given data if scale=4

    Returns:
        ndarray: rescaled mask
    """
    mask = transform.rescale(mask.astype('bool_'), 1/scale, anti_aliasing=False)
    # change mask dtype to bool to ensure the output values are 0 and 1
    # anti_aliasing False otherwise the output might be all 0s
    # the following three lines should be equivalent
    # size0 = int(mask.shape[0] / scale)
    # size1 = int(mask.shape[1] / scale)
    # mask = transform.downscale_local_mean(mask, (size0, size1))
    mask_new = np.zeros(mask.shape)
    mask_new[np.where(mask > 0.5)] = 1
    return mask_new


def cmr_proc(data_path, db, scale, mask_path, mask_id, level):
    """Load image data, masks and perform preprocessing

    Args:
        data_path (str): full path to data file
        db (int): data base id, 2 for 4 chamber and 17 for short axis
        scale (int): sclae factor, e.g. the output will be 1/4 of given data if scale=4
        mask_path (str): full path to image masks
        mask_id (int): index of mask used for preprocessing
        level (int): preprocessing level

    Returns:
        ndarray: preprocessed data
    """
    print('Preprocssing Data Scale: 1/%s, Mask ID: %s, Processing level: %s'
          % (scale, mask_id, level))
    # datadir = os.path.join(basedir, 'DB%s/NoPrs%sDB%s.npy' % (db, scale, db))
    # maskdir = os.path.join(basedir, 'Prep/AllMasks.mat')
    data = np.load(data_path)
    masks = loadmat(mask_path)['masks']
    mask = masks[mask_id-1, db-1]
    mask = scale_cmr_mask(mask, scale)
    data_proc = preproc(data, mask, level)

    return data_proc
