# =============================================================================
# Author: Sina Tabakhi, sina.tabakhi@gmail.com
# =============================================================================

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer

from kale.loaddata.multiomics_datasets import SparseMultiomicsDataset
from kale.pipeline.multiomics_trainer import MultiomicsTrainer


def select_top_features(
    trainer: Trainer,
    model: MultiomicsTrainer,
    dataset: SparseMultiomicsDataset,
    metric: str,
    num_top_feats: int = 30,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute feature importance using an ablation approach (feature masking) and select the top-ranked features.
    Each feature is individually masked, and the performance drop is measured using the specified metric. The
    importance score is defined as the scaled difference between the full and masked model performance.

    Args:
        trainer (Trainer): The PyTorch Lightning Trainer used for evaluation.
        model (MultiomicsTrainer): The trained model compatible with the dataset.
        dataset (SparseMultiomicsDataset): The input dataset created in form of :class:`~torch_geometric.data.Dataset`.
        metric (str): The metric name to evaluate performance drop.
        num_top_feats (int, optional): The number of top features to select. (default: 30)
        verbose (bool, optional): Whether to print Lightning output during testing. (default: ``False``)

    Returns:
        pd.DataFrame: Top features sorted by importance.

    Raises:
        ValueError: If the specified ``metric`` is not found in the test results.
    """
    test_results = trainer.test(model, verbose=verbose)

    if metric not in test_results[0]:
        raise ValueError(f"Metric '{metric}' not found in test results: {list(test_results[0].keys())}")

    metric_full = test_results[0][metric]
    feat_imp_list = []

    for modality_idx in range(dataset.num_modalities):
        modality_data = dataset.get(modality_idx)
        num_feats = modality_data.x.shape[1]
        feat_names = modality_data.feat_names
        imp_scores = np.zeros(num_feats)
        modality_label = np.full(num_feats, modality_idx, dtype=int)

        for feat_idx in range(num_feats):
            # Mask the feature
            feat_data = modality_data.x[:, feat_idx].clone()
            modality_data.x[:, feat_idx] = 0

            # Update dataset and model
            modality_data = dataset.extend_data(modality_data)
            dataset.set(modality_data, modality_idx)
            model.dataset = dataset

            # Re-evaluate model
            test_results = trainer.test(model, verbose=verbose)
            metric_masked = test_results[0][metric]

            # Compute importance
            imp_scores[feat_idx] = (metric_full - metric_masked) * num_feats

            # Restore original feature
            modality_data.x[:, feat_idx] = feat_data
            dataset.set(modality_data, modality_idx)

        modality_data = dataset.extend_data(modality_data)
        dataset.set(modality_data, modality_idx)
        model.dataset = dataset

        df_modality = pd.DataFrame({"feat_name": feat_names, "imp": imp_scores, "omics": modality_label})
        feat_imp_list.append(df_modality)

    df_feat_imp = pd.concat(feat_imp_list, ignore_index=True)
    df_top_feats = df_feat_imp.sort_values(by="imp", ascending=False).iloc[:num_top_feats]

    return df_top_feats
