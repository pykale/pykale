# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================

import neurokit2 as nk
import numpy as np
import torch
from captum.attr import IntegratedGradients
from scipy.ndimage import binary_dilation


def multimodal_signal_image_attribution(
    last_fold_model,
    last_val_loader,
    sample_idx=0,
    signal_threshold=0.70,
    image_threshold=0.7,
    zoom_range=(3, 3.5),
    lead_number=12,
    sampling_rate=500,
    signal_length=10,
):
    """
    Computes model attributions for multimodal (signal + image) input using Integrated Gradients.

    This function selects a sample from the provided validation loader and computes the attributions
    (importance scores) for both signal and image modalities using Captum's Integrated Gradients.
    It returns all relevant arrays and data needed for downstream visualization, including normalized
    attributions, important indices, and segment data for zoomed-in views.

    Parameters
    ----------
    last_fold_model : torch.nn.Module
        Trained multimodal model that accepts both images and signal waveforms as input.
    last_val_loader : DataLoader
        PyTorch DataLoader for the validation dataset. Each batch should yield (image, signal, label).
    sample_idx : int, optional
        Index of the sample in the validation set to interpret (default is 0).
    signal_threshold : float, optional
        Threshold (0-1) to consider signal attributions as important (default is 0.70).
    image_threshold : float, optional
        Threshold (0-1) to consider image attributions as important (default is 0.70).
    zoom_range : tuple of float, optional
        Start and end (in seconds) for zoomed signal visualization window (default is (3, 3.5)).
    lead_number : int, optional
        Number of signal leads (default is 12).
    sampling_rate : int, optional
        Sampling rate of the signal waveform in Hz (default is 500).
    signal_length : int, optional
        signal_length of the signal waveform in Seconds (default is 10).

    Returns
    -------
    dict
        Dictionary containing:
            - label : int
                True class label for the selected sample.
            - predicted_label : int
                Model's predicted class for the sample.
            - predicted_probability : float
                Probability of the predicted class.
            - signal_waveform_np : np.ndarray
                1D numpy array of the processed signal waveform.
            - full_time : np.ndarray
                Time axis (seconds) for the full signal.
            - full_length : int
                Number of time points in the (possibly trimmed) signal.
            - important_indices_full : np.ndarray
                Indices in the full signal considered important by attribution threshold.
            - segment_signal_waveform : np.ndarray
                Zoomed signal segment.
            - zoom_time : np.ndarray
                Time axis (seconds) for the zoomed signal segment.
            - important_indices_zoom : np.ndarray
                Important indices within the zoomed signal segment.
            - zoom_start_sec : float
                Start time (seconds) of the zoomed window.
            - zoom_end_sec : float
                End time (seconds) of the zoomed window.
            - image_np : np.ndarray
                Image as a numpy array.
            - x_pts, y_pts : np.ndarray
                Coordinates of important points in the image (after dilation).
            - importance_pts : np.ndarray
                Attribution values at (x_pts, y_pts).
            - signal_threshold : float
                The threshold used for signal attributions.
            - image_threshold : float
                The threshold used for image attributions.
    """
    batches = list(last_val_loader)
    if not batches:
        raise ValueError("Validation loader is empty. No data to interpret.")

    all_images, all_signals, all_labels = [torch.cat(items) for items in zip(*batches)]

    # --- Select Sample ---
    image = all_images[sample_idx].unsqueeze(0).to(next(last_fold_model.parameters()).device)
    signal = all_signals[sample_idx].unsqueeze(0).to(next(last_fold_model.parameters()).device)
    label = all_labels[sample_idx].item()

    # --- Signal Preprocessing ---
    signal_waveform_1d = all_signals[sample_idx].cpu().numpy().ravel()
    signal_smoothed = nk.ecg_clean(signal_waveform_1d, sampling_rate=sampling_rate)
    signal_smoothed_tensor = (
        torch.tensor(signal_smoothed.copy(), dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(next(last_fold_model.parameters()).device)
    )

    # --- Prediction ---
    last_fold_model.eval()
    with torch.no_grad():
        logits = last_fold_model(image, signal)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        predicted_probability = probabilities[0, predicted_label].item()

    # --- Integrated Gradients ---
    integrated_gradients = IntegratedGradients(last_fold_model)
    image.requires_grad_(True)
    signal.requires_grad_(True)
    attributions, _ = integrated_gradients.attribute(
        inputs=(image, signal_smoothed_tensor),
        target=predicted_label,
        return_convergence_delta=True,
    )
    attributions_image = attributions[0]
    attributions_signal = attributions[1]

    # --- Signal Attribution ---
    attributions_signal_np = attributions_signal.cpu().detach().numpy().squeeze()
    norm_attributions_signal = (attributions_signal_np - attributions_signal_np.min()) / (
        attributions_signal_np.max() - attributions_signal_np.min() + 1e-8
    )
    signal_waveform_np = signal_smoothed_tensor.cpu().detach().numpy().squeeze()
    full_length = min(int(lead_number * sampling_rate * signal_length), len(signal_waveform_np))
    full_time = np.arange(0, full_length) / sampling_rate / lead_number
    important_indices_full = np.where(norm_attributions_signal[:full_length] >= signal_threshold)[0]

    zoom_start = int(zoom_range[0] * int(lead_number * sampling_rate))
    zoom_end = int(zoom_range[1] * int(lead_number * sampling_rate))
    zoom_time = np.arange(zoom_start, zoom_end) / sampling_rate / lead_number
    segment_signal_waveform = signal_waveform_np[zoom_start:zoom_end]
    segment_attributions = norm_attributions_signal[zoom_start:zoom_end]
    important_indices_zoom = np.where(segment_attributions >= signal_threshold)[0]
    zoom_start_sec = zoom_start / sampling_rate / lead_number
    zoom_end_sec = zoom_end / sampling_rate / lead_number

    # --- Image Attribution: Points ---
    attributions_image_np = attributions_image.cpu().detach().numpy().squeeze()
    norm_attributions_image = (attributions_image_np - np.min(attributions_image_np)) / (
        np.max(attributions_image_np) - np.min(attributions_image_np) + 1e-8
    )
    image_np = image.cpu().detach().numpy().squeeze()

    binary_mask = norm_attributions_image >= image_threshold
    dilated_mask = binary_dilation(binary_mask, iterations=1)
    y_pts, x_pts = np.where(dilated_mask)
    importance_pts = norm_attributions_image[y_pts, x_pts]

    return {
        "label": label,
        "predicted_label": predicted_label,
        "predicted_probability": predicted_probability,
        "signal_waveform_np": signal_waveform_np,
        "full_time": full_time,
        "full_length": full_length,
        "important_indices_full": important_indices_full,
        "segment_signal_waveform": segment_signal_waveform,
        "zoom_time": zoom_time,
        "important_indices_zoom": important_indices_zoom,
        "zoom_start_sec": zoom_start_sec,
        "zoom_end_sec": zoom_end_sec,
        "image_np": image_np,
        "x_pts": x_pts,
        "y_pts": y_pts,
        "importance_pts": importance_pts,
        "signal_threshold": signal_threshold,
        "image_threshold": image_threshold,
    }
