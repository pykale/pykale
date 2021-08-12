import os

import numpy as np
import pydicom


def read_dicom_images(dicom_path, sort_instance=True, sort_patient=False):
    """Read dicom images for multiple patients and multiple instances/phases.

    Args:
        dicom_path (str): Path to DICOM images.
        sort_instance (bool, optional): Whether sort images by InstanceNumber (i.e. phase number) for each subject.
            Defaults to True.
        sort_patient (bool, optional): Whether sort subjects' images by PatientID. Defaults to False.

    Returns:
        [array-like]: [description]
    """
    sub_dirs = os.listdir(dicom_path)
    all_ds = []
    sub_ids = []
    for sub_dir in sub_dirs:
        sub_ds = []
        sub_path = os.path.join(dicom_path, sub_dir)
        phase_files = os.listdir(sub_path)
        for phase_file in phase_files:
            dataset = pydicom.dcmread(os.path.join(sub_path, phase_file))
            sub_ds.append(dataset)
        if sort_instance:
            sub_ds.sort(key=lambda x: x.InstanceNumber, reverse=False)
        sub_ids.append(int(sub_ds[0].PatientID))
        all_ds.append(sub_ds)

    if sort_patient:
        all_ds.sort(key=lambda x: int(x[0].PatientID), reverse=False)

    n_sub = len(all_ds)
    n_phase = len(all_ds[0])
    img_shape = all_ds[0][0].pixel_array.shape
    images = np.zeros((n_sub, n_phase,) + img_shape)
    for i in range(n_sub):
        for j in range(n_phase):
            images[i, j, ...] = all_ds[i][j].pixel_array

    return images
