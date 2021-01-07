from abc import ABC

import torch

from kale.pipeline.domain_adapter import Method, BaseMMDLike, BaseDANNLike
import kale.predict.losses as losses


def create_mmd_based_4video(
        method: Method, dataset, image_modality, feature_extractor, task_classifier, **train_params
):
    """MMD-based deep learning methods for domain adaptation: DAN and JAN
    """
    if not method.is_mmd_method():
        raise ValueError(f"Unsupported MMD method: {method}")
    if method is Method.DAN:
        return DANtrainer4Video(
            dataset, image_modality, feature_extractor, task_classifier, method=method, **train_params
        )
    if method is Method.JAN:
        return JANtrainer4Video(
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            method=method,
            kernel_mul=[2.0, 2.0],
            kernel_num=[5, 1],
            **train_params,
        )


class BaseMMDLike4Video(BaseMMDLike):
    def __init__(
            self,
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            kernel_mul=2.0,
            kernel_num=5,
            **base_params,
    ):
        """Common API for MME-based deep learning DA methods: DAN, JAN
        """

        super().__init__(dataset, feature_extractor, task_classifier, kernel_mul, kernel_num, **base_params)
        self.image_modality = image_modality
        self.rgb_feat = self.feat['rgb']
        self.flow_feat = self.feat['flow']

    def forward(self, x):
        if self.feat is not None:
            if self.image_modality in ['rgb', 'flow']:
                if self.rgb_feat is not None:
                    x = self.rgb_feat(x)
                else:
                    x = self.flow_feat(x)
            elif self.image_modality == 'joint':
                x_rgb = self.rgb_feat(x['rgb'])
                x_flow = self.flow_feat(x['flow'])
                x = torch.cat((x_rgb, x_flow), dim=1)
        x = x.view(x.size(0), -1)
        class_output = self.classifier(x)
        return x, class_output

    def compute_loss(self, batch, split_name="V"):
        # if len(batch) == 3:
        #     raise NotImplementedError("MMD does not support semi-supervised setting.")
        if len(batch) == 4:
            (x_s_rgb, y_s), (x_s_flow, y_s_flow), (x_tu_rgb, y_tu), (x_tu_flow, y_tu_flow) = batch
            phi_s, y_hat = self.forward({'rgb': x_s_rgb, 'flow': x_s_flow})
            phi_t, y_t_hat = self.forward({'rgb': x_tu_rgb, 'flow': x_tu_flow})
        elif len(batch) == 2:
            (x_s, y_s), (x_tu, y_tu) = batch
            phi_s, y_hat = self.forward(x_s)
            phi_t, y_t_hat = self.forward(x_tu)
        else:
            raise NotImplementedError("Batch len is {}".format(len(batch)))

        loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)

        mmd = self._compute_mmd(phi_s, phi_t, y_hat, y_t_hat)
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_mmd": mmd,
        }
        return task_loss, mmd, log_metrics


# class BaseDANNLike4Video(BaseDANNLike):
#     def __init__(
#             self,
#             dataset,
#             image_modality,
#             feature_extractor,
#             task_classifier,
#             critic,
#             alpha=1.0,
#             entropy_reg=0.0,  # not used
#             adapt_reg=True,  # not used
#             batch_reweighting=False,  # not used
#             **base_params,
#     ):
#         """Common API for DANN-based methods: DANN, CDAN, CDAN+E, WDGRL, MME, FSDANN
#         """
#
#         super().__init__(dataset, feature_extractor, task_classifier, **base_params)
#         self.image_modality = image_modality
#         self.rgb_feat = self.feat['rgb']
#         self.flow_feat = self.feat['flow']


class DANtrainer4Video(BaseMMDLike4Video):
    """
    This is an implementation of DAN for video data.
    Long, Mingsheng, et al.
    "Learning Transferable Features with Deep Adaptation Networks."
    International Conference on Machine Learning. 2015.
    http://proceedings.mlr.press/v37/long15.pdf
    code based on https://github.com/thuml/Xlearn.
    """

    def __init__(self, dataset, image_modality, feature_extractor, task_classifier, **base_params):
        super().__init__(dataset, image_modality, feature_extractor, task_classifier, **base_params)

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):
        batch_size = int(phi_s.size()[0])
        kernels = losses.gaussian_kernel(
            phi_s, phi_t, kernel_mul=self._kernel_mul, kernel_num=self._kernel_num,
        )
        return losses.compute_mmd_loss(kernels, batch_size)


class JANtrainer4Video(BaseMMDLike4Video):
    """
    This is an implementation of JAN for video data.
    Long, Mingsheng, et al.
    "Deep transfer learning with joint adaptation networks."
    International Conference on Machine Learning, 2017.
    https://arxiv.org/pdf/1605.06636.pdf
    code based on https://github.com/thuml/Xlearn.
    """

    def __init__(self,
                 dataset,
                 image_modality,
                 feature_extractor,
                 task_classifier,
                 kernel_mul=(2.0, 2.0),
                 kernel_num=(5, 1),
                 **base_params,
                 ):
        super().__init__(
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
            **base_params,
        )

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):
        softmax_layer = torch.nn.Softmax(dim=-1)
        source_list = [phi_s, softmax_layer(y_hat)]
        target_list = [phi_t, softmax_layer(y_t_hat)]
        batch_size = int(phi_s.size()[0])

        joint_kernels = None
        for source, target, k_mul, k_num, sigma in zip(
                source_list, target_list, self._kernel_mul, self._kernel_num, [None, 1.68]
        ):
            kernels = losses.gaussian_kernel(
                source, target, kernel_mul=k_mul, kernel_num=k_num, fix_sigma=sigma
            )
            if joint_kernels is not None:
                joint_kernels = joint_kernels * kernels
            else:
                joint_kernels = kernels

        return losses.compute_mmd_loss(joint_kernels, batch_size)


