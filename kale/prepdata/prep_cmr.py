"""
Author: Shuo Zhou, szhou20@sheffield.ac.uk
"""
import logging

import numpy as np
from skimage import exposure, transform


def reg_img_stack(images, coords, dst_id=0):
    """Perform registration for stacked images
    
    Args:
        images (array-like tensor): Input data, shape (dim1, dim2, n_phases, n_samples).
        coords (array-like): Coordinates for registration, shape (n_samples, n_landmarks * 2).
        dst_id (int, optional): Sample index of destination image stack. Defaults to 0.

    Returns:
        array-like: Registered images, shape (dim1, dim2, n_phases, n_samples).
        array-like: Maximum distance of transformed source coordinate to destination coordinate, shape (n_samples,)
    """
    n_phases, n_samples = images.shape[-2:]
    if n_samples != coords.shape[0]:
        error_msg = "The sample size of images and coordinates does not match."
        logging.error(error_msg)
        raise ValueError(error_msg)
    n_landmarks = int(coords.shape[1] / 2)
    dst_coord = coords[dst_id, :]
    dst_coord = dst_coord.reshape((n_landmarks, 2))
    max_dist = np.zeros(n_samples)
    for i in range(n_samples):
        if i == dst_id:
            continue
        else:
            # epts = landmarks.iloc[i, 1:].values
            src_coord = coords[i, :]
            src_coord = src_coord.reshape((n_landmarks, 2))
            idx_valid = np.isnan(src_coord[:, 0])
            tform = transform.estimate_transform(ttype="similarity", src=src_coord[~idx_valid, :],
                                                 dst=dst_coord[~idx_valid, :])
            # forward transform used here, inverse transform used for warp
            src_tform = tform(src_coord[~idx_valid, :])
            dist = np.linalg.norm(src_tform - dst_coord[~idx_valid, :], axis=1)
            max_dist[i] = dist.max()
            for j in range(n_phases):
                src_img = images[..., j, i].copy()
                warped = transform.warp(src_img, inverse_map=tform.inverse, preserve_range=True)
                images[..., j, i] = warped

    return images, max_dist


def rescale_img_stack(images, scale=16):
    """Rescale stacked images by a given factor

    Args:
        images (array-like tensor): Input data, shape (dim1, dim2, n_phases, n_samples).
        scale (int, optional): Scale factor. Defaults to 16.

    Returns:
        array-like tensor: Rescaled images, shape (dim1, dim2, n_phases, n_samples).
    """
    n_phases, n_samples = images.shape[-2:]
    scale_ = 1 / scale
    data_rescale = []
    for i in range(n_samples):
        data_i = []
        for j in range(n_phases):
            img = images[:, :, j, i]
            img_rescale = transform.rescale(img, scale_, preserve_range=True)
            # preserve_range should be true otherwise the output will be normalised values
            data_i.append(img_rescale.reshape(img_rescale.shape + (1,)))
        data_i = np.concatenate(data_i, axis=-1)
        data_rescale.append(data_i.reshape(data_i.shape + (1,)))
    data_rescale = np.concatenate(data_rescale, axis=-1)

    return data_rescale


def mat2gray(img):
    min_ = np.amin(img)
    max_ = np.amax(img)
    diff = max_ - min_
    return (img - min_) / diff


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


def mask_img_stack(images, mask):
    """Masking stacked images by a given mask
    
    Args:
        images (array-like): input image data, shape (dim1, dim2, n_phases, n_subject)
        mask (array-like): mask, shape (dim1, dim2)
    Returns:
        array-like: masked images, shape (dim1, dim2, n_phases, n_subject)
    """
    n_phases, n_samples = images.shape[-2:]
    img_masked = []
    for i in range(n_samples):
        img_masked_i = []
        for j in range(n_phases):
            phase_img = images[:, :, j, i]
            phase_img_masked = np.multiply(phase_img, mask)
            img_masked_i.append(phase_img_masked.reshape(phase_img_masked.shape + (1,)))
        img_masked_i = np.concatenate(img_masked_i, axis=-1)
        img_masked.append(img_masked_i.reshape(img_masked_i.shape + (1,)))
    img_masked = np.concatenate(img_masked, axis=-1)

    return img_masked
