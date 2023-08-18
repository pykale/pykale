from sklearn import metrics

from kale.evaluate.calibration_metrics import compute_calibration


def concord_index(y, y_pred):
    """
    Calculate the Concordance Index (CI), which is a metric to measure the proportion of `concordant pairs
    <https://en.wikipedia.org/wiki/Concordant_pair>`_ between real and
    predict values.

    Args:
        y (array): real values.
        y_pred (array): predicted values.
    """
    total_loss = 0
    pair = 0
    for i in range(1, len(y)):
        for j in range(0, i):
            if i is not j:
                if y[i] > y[j]:
                    pair += 1
                    total_loss += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])

    if pair:
        return total_loss / pair
    else:
        return 0


def performance_metrices(truth, pred, score):

    # Get the confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(truth, pred).ravel()

    # Compute basic metrics using sklearn
    accuracy = metrics.accuracy_score(truth, pred)
    precision = metrics.precision_score(truth, pred, zero_division=0)
    recall = metrics.recall_score(truth, pred, zero_division=0)
    f1_score = metrics.f1_score(truth, pred, zero_division=0)

    # Specificity and NPV should be calculated as you did
    specificity = tn / (tn + fp)
    if (tn + fn) == 0:
        npv = 0.0
    else:
        npv = tn / (tn + fn)

    # Compute AUROC and AUPRC
    if score.shape[1] == 2:  # Binary classification
        auroc = metrics.roc_auc_score(truth, score[:, 1])
        # For binary classification, AUPRC can be calculated using average_precision_score directly
        auprc = metrics.average_precision_score(truth, score[:, 1])
    else:  # Multi-class
        auroc = metrics.roc_auc_score(truth, score, multi_class="ovr")
        # For multi-class, it's the average AP across all classes.
        auprc = metrics.average_precision_score(truth, score, multi_class="ovr")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Specificity": specificity,
        "NVP": npv,
        "AUROC": auroc,
        "AUPRC": auprc,
    }


def benchmark_evalaution(truth, pred, score):

    p_metrics = performance_metrices(truth, pred, score)

    c_metrics = compute_calibration(truth, pred, score)

    merged_metrics = {**p_metrics, **c_metrics}

    return merged_metrics


"""
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
"""
