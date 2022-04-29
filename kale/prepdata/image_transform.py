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
            transform_aug = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256)])
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


def reg_img_stack(images, coords, target_coords):
    """Registration for stacked images

    Args:
        images (list): Input data, where each sample in shape (n_phases, dim1, dim2).
        coords (array-like): Coordinates for registration, shape (n_samples, n_landmarks * 2).
        target_coords (array-like): Target coordinates for registration.

    Returns:
        list: Registered images, each sample in the list in shape (n_phases, dim1, dim2).
        array-like: Maximum distance of transformed source coordinates to destination coordinates, shape (n_samples,)
    """
    n_samples = len(images)
    if n_samples != coords.shape[0]:
        error_msg = "The sample size of images and coordinates does not match."
        logging.error(error_msg)
        raise ValueError(error_msg)
    n_landmarks = int(coords.shape[1] / 2)

    target_coords = target_coords.reshape((n_landmarks, 2))
    max_dist = np.zeros(n_samples)
    for i in range(n_samples):
        src_coord = coords[i, :]
        src_coord = src_coord.reshape((n_landmarks, 2))
        idx_valid = np.isnan(src_coord[:, 0])
        tform = estimate_transform(ttype="similarity", src=src_coord[~idx_valid, :], dst=target_coords[~idx_valid, :])
        # forward transform used here, inverse transform used for warp
        src_tform = tform(src_coord[~idx_valid, :])
        dists = np.linalg.norm(src_tform - target_coords[~idx_valid, :], axis=1)
        max_dist[i] = np.max(dists)
        n_phases = images[i].shape[0]
        for j in range(n_phases):
            src_img = images[i][j, ...].copy()
            warped = warp(src_img, inverse_map=tform.inverse, preserve_range=True)
            images[i][j, ...] = warped

    return images, max_dist


def rescale_img_stack(images, scale=0.5):
    """Rescale stacked images by a given factor

    Args:
        images (list): Input data list, where each sample in shape (n_phases, dim1, dim2).
        scale (float, optional): Scale factor. Defaults to 0.5.

    Returns:
        list: Rescaled images, each sample in the list in shape (n_phases, dim1 * scale, dim2 * scale).
    """
    n_samples = len(images)
    # n_phases = images.shape[:2]
    images_rescale = []
    for i in range(n_samples):
        stack_i = []
        n_phases = images[i].shape[0]
        for j in range(n_phases):
            img = images[i][j, ...]
            img_rescale = rescale(img, scale, preserve_range=True)
            # preserve_range should be true otherwise the output will be normalised values
            stack_i.append(img_rescale.reshape((1,) + img_rescale.shape))
        stack_i = np.concatenate(stack_i, axis=0)
        images_rescale.append(stack_i)

    return images_rescale


def mask_img_stack(images, mask):
    """Masking stacked images by a given mask

    Args:
        images (list): Input image data, where each sample in shape (n_phases, dim1, dim2).
        mask (array-like): mask, shape (dim1, dim2).
    Returns:
        list: masked images, each sample in the list in shape (n_phases, dim1, dim2).
    """
    n_samples = len(images)
    for i in range(n_samples):
        n_phases = images[i].shape[0]
        for j in range(n_phases):
            images[i][j, ...] = np.multiply(images[i][j, ...], mask)

    return images


def normalize_img_stack(images):
    """Normalize pixel values to (0, 1) for stacked images.

    Args:
        images (list): Input data, where each sample in shape (n_phases, dim1, dim2).

    Returns:
        list: Normalized images, each sample in the list in shape (n_phases, dim1, dim2).
    """
    n_samples = len(images)
    for i in range(n_samples):
        n_phases = images[i].shape[0]
        for j in range(n_phases):
            img = images[i][j, ...]
            images[i][j, ...] = (img - np.min(img)) / (np.max(img) - np.min(img))

    return images
