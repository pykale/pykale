"""This module implements a Multimodal Deep Learning (MMDL) classifier. The MMDL classifier uses separate encoders for each modality, a fusion technique to combine the modalities, and a classifier head for final prediction.
   This approach enables the extraction of intricate patterns from multi-modal data, leading to improved prediction accuracy.
   The module is designed to be flexible, allowing for easy integration of various encoders, fusion methods, and classifiers. This ensures that the system can adapt to various multi-modal tasks and datasets, enhancing its utility in diverse multi-modal learning scenarios.
   Reference: https://github.com/pliang279/MultiBench/blob/main/training_structures/Supervised_Learning.py
"""

from torch import nn


class MultimodalDeepLearning(nn.Module):
    """The MMDL classifier is built from three components:
       1. Encoders: A list of PyTorch `nn.Module` encoders, with one encoder per modality. Each encoder is responsible for transforming the raw input of a single modality into a high-level representation.
       2. Fusion Module: A PyTorch `nn.Module` that merges the high-level representations from each modality into a single representation. This fusion can be performed in various ways, such as concatenation or more complex fusion methods like tensor-based fusion.
       3. Head/Classifier: A PyTorch `nn.Module` that takes the fused representation and outputs a class prediction. This is typically a feedforward neural network with one output node for binary classification tasks, or `n` output nodes for multi-class classification tasks where `n` is the number of classes.
       During the forward pass, the model first applies each encoder to the corresponding modality to obtain high-level representations. It then fuses these representations using the fusion module and passes the fused representation to the classifier to get the final output.
    Args:
        encoders (List): List of nn.Module encoders, one per modality.
        fusion (nn.Module): Fusion module
        head (nn.Module): Classifier module
        has_padding (bool, optional): Whether input has padding or not. Defaults to False.
    """

    def __init__(self, encoders, fusion, head):
        super(MultimodalDeepLearning, self).__init__()
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
