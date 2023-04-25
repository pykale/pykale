"""Implements Multimodal Deep Learning (MMDL) classifier.
References: https://github.com/pliang279/MultiBench/blob/main/training_structures/Supervised_Learning.py
"""
from torch import nn


class MultiModalDeepLearning(nn.Module):
    """Instantiate MultiModalDeepLearning Module
    Args:
    encoders (List): List of nn.Module encoders, one per modality.
    fusion (nn.Module): Fusion module
    head (nn.Module): Classifier module
    has_padding (bool, optional): Whether input has padding or not. Defaults to False.
    """

    def __init__(self, encoders, fusion, head):
        super(MultiModalDeepLearning, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fusion_module = fusion
        self.classifier = head
        self.fusion_output = None
        self.modalities_reps = []

    def forward(self, inputs):
        modality_outputs = []
        for i in range(len(inputs)):
            modality_outputs.append(self.encoders[i](inputs[i]))
        self.modalities_reps = modality_outputs
        fused_output = self.fusion_module(modality_outputs)
        self.fusion_output = fused_output
        if type(fused_output) is tuple:
            fused_output = fused_output[0]
        return self.classifier(fused_output)
