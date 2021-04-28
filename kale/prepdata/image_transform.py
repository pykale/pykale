"""
Preprocessing of image datasets, i.e., transforms, from
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/preprocessing.py

References for processing stacked images:
    Swift, A. J., Lu, H., Uthoff, J., Garg, P., Cogliano, M., Taylor, J., ... & Kiely, D. G. (2020). A machine
    learning cardiac magnetic resonance approach to extract disease features and automate pulmonary arterial
    hypertension diagnosis. European Heart Journal-Cardiovascular Imaging.
"""
import logging

import numpy as np
import torchvision.transforms as transforms
from skimage.transform import estimate_transform, rescale, warp


def get_transform(kind, augment=False):
    """
    Define transforms (for commonly used datasets)

    Args:
        kind ([type]): the dataset (transformation) name
        augment (bool, optional): whether to do data augmentation (random crop and flipping). Defaults to False.
            (Not implemented for digits yet.)

    """
    if kind == "mnist32":
        transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
    elif kind == "mnist32rgb":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif kind == "usps32":
        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
    elif kind == "usps32rgb":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(32),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif kind == "mnistm":
        transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    elif kind == "svhn":
        transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    elif kind == "cifar":
        if augment:
            transform_aug = transforms.Compose(
                [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            )
        else:
            transform_aug = transforms.Compose([])
        transform = transforms.Compose(
            [
                transform_aug,
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    elif kind == "office":
        if augment:
            transform_aug = transforms.Compose(
                [transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
            )
        else:
            transform_aug = transforms.Compose([transforms.Resize(256)])
        transform = transforms.Compose(
            [
                transform_aug,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        raise ValueError(f"Unknown transform kind '{kind}'")
    return transform


def reg_img_stack(images, coords, dst_id=0):
    """Registration for stacked images

    Args:
        images (array-like tensor): Input data, shape (dim1, dim2, n_phases, n_samples).
        coords (array-like): Coordinates for registration, shape (n_samples, n_landmarks * 2).
        dst_id (int, optional): Sample index of destination image stack. Defaults to 0.

    Returns:
        array-like: Registered images, shape (dim1, dim2, n_phases, n_samples).
        array-like: Maximum distance of transformed source coordinates to destination coordinates, shape (n_samples,)
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
            src_coord = coords[i, :]
            src_coord = src_coord.reshape((n_landmarks, 2))
            idx_valid = np.isnan(src_coord[:, 0])
            tform = estimate_transform(ttype="similarity", src=src_coord[~idx_valid, :], dst=dst_coord[~idx_valid, :])
            # forward transform used here, inverse transform used for warp
            src_tform = tform(src_coord[~idx_valid, :])
            dists = np.linalg.norm(src_tform - dst_coord[~idx_valid, :], axis=1)
            max_dist[i] = np.max(dists)
            for j in range(n_phases):
                src_img = images[..., j, i].copy()
                warped = warp(src_img, inverse_map=tform.inverse, preserve_range=True)
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
    images_rescale = []
    for i in range(n_samples):
        stack_i = []
        for j in range(n_phases):
            img = images[:, :, j, i]
            img_rescale = rescale(img, scale_, preserve_range=True)
            # preserve_range should be true otherwise the output will be normalised values
            stack_i.append(img_rescale.reshape(img_rescale.shape + (1,)))
        stack_i = np.concatenate(stack_i, axis=-1)
        images_rescale.append(stack_i.reshape(stack_i.shape + (1,)))
    images_rescale = np.concatenate(images_rescale, axis=-1)

    return images_rescale


def mask_img_stack(images, mask):
    """Masking stacked images by a given mask

    Args:
        images (array-like): input image data, shape (dim1, dim2, n_phases, n_subject)
        mask (array-like): mask, shape (dim1, dim2)
    Returns:
        array-like tensor: masked images, shape (dim1, dim2, n_phases, n_subject)
    """
    n_phases, n_samples = images.shape[-2:]
    for i in range(n_samples):
        for j in range(n_phases):
            images[:, :, j, i] = np.multiply(images[:, :, j, i], mask)

    return images
