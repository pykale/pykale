import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import calibration_curve, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def get_predictions_targets(model, test_loader, apply_softmax=False):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if apply_softmax:
                output = torch.nn.functional.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    return all_preds, all_targets


def test_robustness(model, test_loader, noise_factor):
    model.eval()
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            noise = noise_factor * torch.randn(*data.shape).to(device)
            data += noise
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += target.size(0)
    accuracy = correct / total_samples
    return accuracy


def test_complementarity(model, image_loader, audio_loader):
    model.eval()
    total_samples = 0
    image_correct, audio_correct = 0, 0
    with torch.no_grad():
        for ((image_data, _), (audio_data, target)) in zip(image_loader, audio_loader):
            image_data, audio_data, target = image_data.to(device), audio_data.to(device), target.to(device)
            image_output = model(image_data, torch.zeros_like(audio_data))
            audio_output = model(torch.zeros_like(image_data), audio_data)
            total_samples += target.size(0)
            image_correct += image_output.argmax(dim=1).eq(target.view_as(image_output.argmax(dim=1))).sum().item()
            audio_correct += audio_output.argmax(dim=1).eq(target.view_as(audio_output.argmax(dim=1))).sum().item()
    image_accuracy = image_correct / total_samples
    audio_accuracy = audio_correct / total_samples
    return image_accuracy, audio_accuracy


def test_class_performance(model, test_loader):
    all_preds, all_targets = get_predictions_targets(model, test_loader)
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    for i in range(len(all_targets)):
        target_i = all_targets[i]
        class_correct[target_i] += all_preds[i] == all_targets[i]
        class_total[target_i] += 1
    class_accuracy = [correct / total for correct, total in zip(class_correct, class_total)]
    return class_accuracy


def test_inference_time(model, test_loader):
    model.eval()
    inference_time = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            start_time = time.time()
            output = model(data)
            inference_time.append(time.time() - start_time)
    avg_inference_time = sum(inference_time) / len(inference_time)
    return avg_inference_time


def visualize_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:
            plt.figure(figsize=(15, 3))
            plt.title(name)
            plt.imshow(param.data.cpu().numpy(), cmap="viridis")
            plt.colorbar()
            plt.show()


def test_fairness(model, test_loader, sensitive_attribute_idx):
    model.eval()
    fairness_dict = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            for i in range(target.size()[0]):
                sensitive_attribute = data[i][sensitive_attribute_idx].item()
                if sensitive_attribute not in fairness_dict:
                    fairness_dict[sensitive_attribute] = [0, 0]  # [correct, total]
                fairness_dict[sensitive_attribute][1] += 1
                if pred[i].item() == target[i].item():
                    fairness_dict[sensitive_attribute][0] += 1
    fairness_dict = {k: v[0] / v[1] for k, v in fairness_dict.items()}  # compute accuracy
    return fairness_dict


def test_calibration(model, test_loader):
    model_probs, true_labels = get_predictions_targets(model, test_loader, apply_softmax=True)
    fraction_of_positives, mean_predicted_value = calibration_curve(true_labels, model_probs, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    plt.plot([0, 1], [0, 1], "k--")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.show()


def test_out_of_distribution_detection(model, in_distribution_loader, out_of_distribution_loader):
    in_distribution_probs, _ = get_predictions_targets(model, in_distribution_loader, apply_softmax=True)
    out_of_distribution_probs, _ = get_predictions_targets(model, out_of_distribution_loader, apply_softmax=True)
    in_distribution_probs = np.array(in_distribution_probs)
    out_of_distribution_probs = np.array(out_of_distribution_probs)
    accuracy_in_distribution = np.mean(in_distribution_probs > 0.5)
    accuracy_out_of_distribution = np.mean(out_of_distribution_probs <= 0.5)
    return accuracy_in_distribution, accuracy_out_of_distribution


def test_metrics(model, test_loader):
    all_preds, all_targets = get_predictions_targets(model, test_loader)
    precision = precision_score(all_targets, all_preds, average="weighted")
    recall = recall_score(all_targets, all_preds, average="weighted")
    f1 = f1_score(all_targets, all_preds, average="weighted")
    auc = roc_auc_score(all_targets, all_preds, average="weighted", multi_class="ovr")
    return precision, recall, f1, auc


def test_sensitivity_specificity(model, test_loader):
    all_preds, all_targets = get_predictions_targets(model, test_loader)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity
