# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Domain adaptation systems (pipelines) for video data, e.g., for action recognition.
Most are inherited from kale.pipeline.domain_adapter.
"""
import math
import time

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn.init import normal_, constant_, kaiming_normal_

import kale.predict.losses as losses
from kale.embed.video_trn import TemporalAttention, TRNRelationModuleMultiScale, TRNRelationModule, TCL
from kale.loaddata.video_access import get_class_type, get_image_modality
from kale.pipeline.domain_adapter import (
    BaseAdaptTrainer,
    BaseMMDLike,
    CDANtrainer,
    DANNtrainer,
    get_aggregated_metrics,
    get_aggregated_metrics_from_dict,
    get_metrics_from_parameter_dict,
    GradReverse,
    Method,
    set_requires_grad,
    WDGRLtrainer,
)


# from kale.utils.logger import save_results_to_json


def create_mmd_based_video(
        method: Method, dataset, image_modality, feature_extractor, task_classifier, input_type, class_type,
        **train_params
):
    """MMD-based deep learning methods for domain adaptation on video data: DAN and JAN"""
    if not method.is_mmd_method():
        raise ValueError(f"Unsupported MMD method: {method}")
    if method is Method.DAN:
        return DANTrainerVideo(
            dataset=dataset,
            image_modality=image_modality,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            method=method,
            input_type=input_type,
            class_type=class_type,
            **train_params,
        )
    if method is Method.JAN:
        return JANTrainerVideo(
            dataset=dataset,
            image_modality=image_modality,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            method=method,
            input_type=input_type,
            class_type=class_type,
            kernel_mul=[2.0, 2.0],
            kernel_num=[5, 1],
            **train_params,
        )


def create_dann_like_video(
        method: Method,
        dataset,
        image_modality,
        feature_extractor,
        task_classifier,
        critic,
        input_type,
        class_type,
        **train_params,
):
    """DANN-based deep learning methods for domain adaptation on video data: DANN, CDAN, CDAN+E"""

    # # Uncomment for later work.
    # # Set up a new create_fewshot_trainer for video data based on original one in `domain_adapter.py`
    # if dataset.is_semi_supervised():
    #     return create_fewshot_trainer_4video(
    #         method, dataset, feature_extractor, task_classifier, critic, **train_params
    #     )

    if method.is_dann_method():
        alpha = 0 if method is Method.Source else 1
        return DANNTrainerVideo(
            alpha=alpha,
            image_modality=image_modality,
            dataset=dataset,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
            input_type=input_type,
            class_type=class_type,
            **train_params,
        )
    elif method.is_cdan_method():
        return CDANTrainerVideo(
            dataset=dataset,
            image_modality=image_modality,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
            input_type=input_type,
            class_type=class_type,
            use_entropy=method is Method.CDAN_E,
            **train_params,
        )
    elif method.is_ta3n_method():
        return TA3NTrainerVideo(
            image_modality=image_modality,
            dataset=dataset,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
            input_type=input_type,
            class_type=class_type,
            **train_params,
        )
    elif method is Method.WDGRL:
        return WDGRLTrainerVideo(
            dataset=dataset,
            image_modality=image_modality,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
            input_type=input_type,
            class_type=class_type,
            **train_params,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")


class BaseAdaptTrainerVideo(BaseAdaptTrainer):
    def train_dataloader(self):
        dataloader, target_batch_size = self._dataset.get_domain_loaders(split="train", batch_size=self._batch_size)
        self._nb_training_batches = len(dataloader)
        self._target_batch_size = target_batch_size
        return dataloader

    def val_dataloader(self):
        dataloader, target_batch_size = self._dataset.get_domain_loaders(split="valid", batch_size=self._batch_size)
        self._target_batch_size = target_batch_size
        return dataloader

    def test_dataloader(self):
        dataloader, target_batch_size = self._dataset.get_domain_loaders(split="test", batch_size=self._batch_size)
        # dataloader, target_batch_size = self._dataset.get_domain_loaders(split="test", batch_size=500)
        self._target_batch_size = target_batch_size
        return dataloader

    def training_step(self, batch, batch_nb):
        # print("tr src{} tgt{}".format(len(batch[0][2]), len(batch[1][2])))

        self._update_batch_epoch_factors(batch_nb)

        task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="T")
        if self.current_epoch < self._init_epochs:
            loss = task_loss
        else:
            # loss = task_loss
            loss = task_loss + self.lamb_da * adv_loss

        log_metrics = get_aggregated_metrics_from_dict(log_metrics)
        log_metrics.update(get_metrics_from_parameter_dict(self.get_parameters_watch_list(), loss.device))
        log_metrics["T_total_loss"] = loss
        log_metrics["T_adv_loss"] = adv_loss
        log_metrics["T_task_loss"] = task_loss

        for key in log_metrics:
            self.log(key, log_metrics[key])
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        if self.verb and not self.noun:
            metrics_to_log = (
                "val_loss",
                "val_task_loss",
                "val_adv_loss",
                "V_source_acc",
                "V_source_top1_acc",
                "V_source_top5_acc",
                "V_target_acc",
                "V_target_top1_acc",
                "V_target_top5_acc",
                "V_source_domain_acc",
                "V_target_domain_acc",
                "V_domain_acc",
            )
        elif self.verb and self.noun:
            metrics_to_log = (
                "val_loss",
                "val_task_loss",
                "val_adv_loss",
                "V_verb_source_acc",
                "V_verb_source_top1_acc",
                "V_verb_source_top5_acc",
                "V_noun_source_acc",
                "V_noun_source_top1_acc",
                "V_noun_source_top5_acc",
                "V_verb_target_acc",
                "V_verb_target_top1_acc",
                "V_verb_target_top5_acc",
                "V_noun_target_acc",
                "V_noun_target_top1_acc",
                "V_noun_target_top5_acc",
                "V_action_source_top1_acc",
                "V_action_source_top5_acc",
                "V_action_target_top1_acc",
                "V_action_target_top5_acc",
                "V_domain_acc",
            )
        return self._validation_epoch_end(outputs, metrics_to_log)

    def test_epoch_end(self, outputs):
        if self.verb and not self.noun:
            metrics_at_test = (
                "test_loss",
                "Te_source_acc",
                "Te_source_top1_acc",
                "Te_source_top5_acc",
                "Te_target_acc",
                "Te_target_top1_acc",
                "Te_target_top5_acc",
                "Te_domain_acc",
            )
        elif self.verb and self.noun:
            metrics_at_test = (
                "test_loss",
                "Te_verb_source_acc",
                "Te_verb_source_top1_acc",
                "Te_verb_source_top5_acc",
                "Te_noun_source_acc",
                "Te_noun_source_top1_acc",
                "Te_noun_source_top5_acc",
                "Te_verb_target_acc",
                "Te_verb_target_top1_acc",
                "Te_verb_target_top5_acc",
                "Te_noun_target_acc",
                "Te_noun_target_top1_acc",
                "Te_noun_target_top5_acc",
                "Te_action_source_top1_acc",
                "Te_action_source_top5_acc",
                "Te_action_target_top1_acc",
                "Te_action_target_top5_acc",
                "Te_domain_acc",
            )

        # Uncomment to save output to json file for EPIC UDA 2021 challenge.(3/3)
        # save_results_to_json(
        #     self.y_hat, self.y_t_hat, self.s_id, self.tu_id, self.y_hat_noun, self.y_t_hat_noun, self.verb, self.noun
        # )
        log_dict = get_aggregated_metrics(metrics_at_test, outputs)

        for key in log_dict:
            self.log(key, log_dict[key], prog_bar=True)

    def concatenate_feature(self, x_rgb, x_flow, x_audio):
        if self.rgb:
            if self.flow:
                if self.audio:  # For all inputs
                    x = torch.cat((x_rgb, x_flow, x_audio), dim=-1)
                else:  # For joint(rgb+flow) input
                    x = torch.cat((x_rgb, x_flow), dim=-1)
            else:
                if self.audio:  # For rgb+audio input
                    x = torch.cat((x_rgb, x_audio), dim=-1)
                else:  # For rgb input
                    x = x_rgb
        else:
            if self.flow:
                if self.audio:  # For flow+audio input
                    x = torch.cat((x_flow, x_audio), dim=-1)
                else:  # For flow input
                    x = x_flow
            else:  # For audio input
                x = x_audio
        return x

    def get_inputs_from_batch(self, batch):
        # _s refers to source, _tu refers to unlabeled target
        x_s_rgb = x_tu_rgb = x_s_flow = x_tu_flow = x_s_audio = x_tu_audio = None

        if self.rgb:
            if self.flow:
                if self.audio:  # For all inputs
                    (
                        (x_s_rgb, y_s, s_id),
                        (x_s_flow, y_s_flow, _),
                        (x_s_audio, y_s_audio, _),
                        (x_tu_rgb, y_tu, tu_id),
                        (x_tu_flow, y_tu_flow, _),
                        (x_tu_audio, y_tu_audio, _),
                    ) = batch
                else:  # For joint(rgb+flow) input
                    (
                        (x_s_rgb, y_s, s_id),
                        (x_s_flow, y_s_flow, _),
                        (x_tu_rgb, y_tu, tu_id),
                        (x_tu_flow, y_tu_flow, _),
                    ) = batch
            else:
                if self.audio:  # For rgb+audio input
                    (
                        (x_s_rgb, y_s, s_id),
                        (x_s_audio, y_s_audio, _),
                        (x_tu_rgb, y_tu, tu_id),
                        (x_tu_audio, y_tu_audio, _),
                    ) = batch
                else:  # For rgb input
                    (x_s_rgb, y_s, s_id), (x_tu_rgb, y_tu, tu_id) = batch
        else:
            if self.flow:
                if self.audio:  # For flow+audio input
                    (
                        (x_s_flow, y_s, s_id),
                        (x_s_audio, y_s_audio, _),
                        (x_tu_flow, y_tu, tu_id),
                        (x_tu_audio, y_tu_audio, _),
                    ) = batch
                else:  # For flow input
                    (x_s_flow, y_s, s_id), (x_tu_flow, y_tu, tu_id) = batch
            else:  # For audio input
                (x_s_audio, y_s, s_id), (x_tu_audio, y_tu, tu_id) = batch

        return x_s_rgb, x_tu_rgb, x_s_flow, x_tu_flow, x_s_audio, x_tu_audio, y_s, y_tu, s_id, tu_id

    def get_loss_log_metrics(self, split_name, y_hat, y_t_hat, y_s, y_tu, dok):
        if self.verb and not self.noun:
            loss_cls, ok_src = losses.cross_entropy_logits(y_hat[0], y_s[0])
            _, ok_tgt = losses.cross_entropy_logits(y_t_hat[0], y_tu[0])
            prec1_src, prec5_src = losses.topk_accuracy(y_hat[0], y_s[0], topk=(1, 5))
            prec1_tgt, prec5_tgt = losses.topk_accuracy(y_t_hat[0], y_tu[0], topk=(1, 5))
            task_loss = loss_cls

            log_metrics = {
                f"{split_name}_source_acc": ok_src,
                f"{split_name}_target_acc": ok_tgt,
                f"{split_name}_source_top1_acc": prec1_src,
                f"{split_name}_source_top5_acc": prec5_src,
                f"{split_name}_target_top1_acc": prec1_tgt,
                f"{split_name}_target_top5_acc": prec5_tgt,
                f"{split_name}_domain_acc": dok,
            }

        elif self.verb and self.noun:
            loss_cls_verb, ok_src_verb = losses.cross_entropy_logits(y_hat[0], y_s[0])
            loss_cls_noun, ok_src_noun = losses.cross_entropy_logits(y_hat[1], y_s[1])
            _, ok_tgt_verb = losses.cross_entropy_logits(y_t_hat[0], y_tu[0])
            _, ok_tgt_noun = losses.cross_entropy_logits(y_t_hat[1], y_tu[1])

            prec1_src_verb, prec5_src_verb = losses.topk_accuracy(y_hat[0], y_s[0], topk=(1, 5))
            prec1_src_noun, prec5_src_noun = losses.topk_accuracy(y_hat[1], y_s[1], topk=(1, 5))
            prec1_src_action, prec5_src_action = losses.multitask_topk_accuracy(
                (y_hat[0], y_hat[1]), (y_s[0], y_s[1]), topk=(1, 5)
            )
            prec1_tgt_verb, prec5_tgt_verb = losses.topk_accuracy(y_t_hat[0], y_tu[0], topk=(1, 5))
            prec1_tgt_noun, prec5_tgt_noun = losses.topk_accuracy(y_t_hat[1], y_tu[1], topk=(1, 5))
            prec1_tgt_action, prec5_tgt_action = losses.multitask_topk_accuracy(
                (y_t_hat[0], y_t_hat[1]), (y_tu[0], y_tu[1]), topk=(1, 5)
            )

            task_loss = loss_cls_verb + loss_cls_noun

            log_metrics = {
                f"{split_name}_verb_source_acc": ok_src_verb,
                f"{split_name}_noun_source_acc": ok_src_noun,
                f"{split_name}_verb_target_acc": ok_tgt_verb,
                f"{split_name}_noun_target_acc": ok_tgt_noun,
                f"{split_name}_verb_source_top1_acc": prec1_src_verb,
                f"{split_name}_verb_source_top5_acc": prec5_src_verb,
                f"{split_name}_noun_source_top1_acc": prec1_src_noun,
                f"{split_name}_noun_source_top5_acc": prec5_src_noun,
                f"{split_name}_action_source_top1_acc": prec1_src_action,
                f"{split_name}_action_source_top5_acc": prec5_src_action,
                f"{split_name}_verb_target_top1_acc": prec1_tgt_verb,
                f"{split_name}_verb_target_top5_acc": prec5_tgt_verb,
                f"{split_name}_noun_target_top1_acc": prec1_tgt_noun,
                f"{split_name}_noun_target_top5_acc": prec5_tgt_noun,
                f"{split_name}_action_target_top1_acc": prec1_tgt_action,
                f"{split_name}_action_target_top5_acc": prec5_tgt_action,
                f"{split_name}_domain_acc": dok,
            }
        else:
            raise ValueError("Invalid class type option")
        return task_loss, log_metrics


class BaseMMDLikeVideo(BaseAdaptTrainerVideo, BaseMMDLike):
    def __init__(
            self,
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            class_type,
            input_type,
            kernel_mul=2.0,
            kernel_num=5,
            **base_params,
    ):
        """Common API for MME-based domain adaptation on video data: DAN, JAN"""

        super().__init__(dataset, feature_extractor, task_classifier, kernel_mul, kernel_num, **base_params)
        self.image_modality = image_modality
        self.rgb, self.flow, self.audio = get_image_modality(self.image_modality)
        self.class_type = class_type
        self.verb, self.noun = get_class_type(self.class_type)
        self.rgb_feat = self.feat["rgb"]
        self.flow_feat = self.feat["flow"]
        self.audio_feat = self.feat["audio"]
        self.input_type = input_type

    def forward(self, x):
        if self.feat is not None:
            x_rgb = x_flow = x_audio = None

            # For joint input, both two ifs are used
            if self.rgb:
                x_rgb = self.rgb_feat(x["rgb"])
                x_rgb = x_rgb.view(x_rgb.size(0), -1)
            if self.flow:
                x_flow = self.flow_feat(x["flow"])
                x_flow = x_flow.view(x_flow.size(0), -1)
            if self.audio:
                x_audio = self.audio_feat(x["audio"])
                x_audio = x_audio.view(x_audio.size(0), -1)

            x = self.concatenate_feature(x_rgb, x_flow, x_audio)
            class_output = self.classifier(x)
            return [x_rgb, x_flow, x_audio], class_output

    def compute_loss(self, batch, split_name="V"):
        # _s refers to source, _tu refers to unlabeled target
        (
            x_s_rgb,
            x_tu_rgb,
            x_s_flow,
            x_tu_flow,
            x_s_audio,
            x_tu_audio,
            y_s,
            y_tu,
            s_id,
            tu_id,
        ) = self.get_inputs_from_batch(batch)

        [phi_s_rgb, phi_s_flow, phi_s_audio], y_hat = self.forward(
            {"rgb": x_s_rgb, "flow": x_s_flow, "audio": x_s_audio}
        )
        [phi_t_rgb, phi_t_flow, phi_t_audio], y_t_hat = self.forward(
            {"rgb": x_tu_rgb, "flow": x_tu_flow, "audio": x_tu_audio}
        )

        if self.rgb:
            if self.verb and not self.noun:
                mmd_rgb = self._compute_mmd(phi_s_rgb, phi_t_rgb, y_hat[0], y_t_hat[0])
            elif self.verb and self.noun:
                mmd_rgb_verb = self._compute_mmd(phi_s_rgb, phi_t_rgb, y_hat[0], y_t_hat[0])
                mmd_rgb_noun = self._compute_mmd(phi_s_rgb, phi_t_rgb, y_hat[1], y_t_hat[1])
                mmd_rgb = mmd_rgb_verb + mmd_rgb_noun
        if self.flow:
            if self.verb and not self.noun:
                mmd_flow = self._compute_mmd(phi_s_flow, phi_t_flow, y_hat[0], y_t_hat[0])
            elif self.verb and self.noun:
                mmd_flow_verb = self._compute_mmd(phi_s_flow, phi_t_flow, y_hat[0], y_t_hat[0])
                mmd_flow_noun = self._compute_mmd(phi_s_flow, phi_t_flow, y_hat[1], y_t_hat[1])
                mmd_flow = mmd_flow_verb + mmd_flow_noun
        if self.audio:
            if self.verb and not self.noun:
                mmd_audio = self._compute_mmd(phi_s_audio, phi_t_audio, y_hat[0], y_t_hat[0])
            elif self.verb and self.noun:
                mmd_audio_verb = self._compute_mmd(phi_s_audio, phi_t_audio, y_hat[0], y_t_hat[0])
                mmd_audio_noun = self._compute_mmd(phi_s_audio, phi_t_audio, y_hat[1], y_t_hat[1])
                mmd_audio = mmd_audio_verb + mmd_audio_noun

        if self.rgb:
            if self.flow:
                if self.audio:  # For all inputs
                    mmd = mmd_rgb + mmd_flow + mmd_audio
                else:  # For joint(rgb+flow) input
                    mmd = mmd_rgb + mmd_flow
            else:
                if self.audio:  # For rgb+audio input
                    mmd = mmd_audio
                else:  # For rgb input
                    mmd = mmd_rgb
        else:
            if self.flow:
                if self.audio:  # For flow+audio input
                    mmd = mmd_audio
                else:  # For flow input
                    mmd = mmd_flow
            else:  # For audio input
                mmd = mmd_audio

        # Uncomment when checking whether rgb & flow labels are equal.
        # print('rgb_s:{}, flow_s:{}, rgb_f:{}, flow_f:{}'.format(y_s, y_s_flow, y_tu, y_tu_flow))
        # print('equal: {}/{}'.format(torch.all(torch.eq(y_s, y_s_flow)), torch.all(torch.eq(y_tu, y_tu_flow))))

        task_loss, log_metrics = self.get_loss_log_metrics(split_name, y_hat, y_t_hat, y_s, y_tu, mmd)

        return task_loss, mmd, log_metrics


class DANTrainerVideo(BaseMMDLikeVideo):
    """This is an implementation of DAN for video data."""

    def __init__(self, dataset, image_modality, feature_extractor, task_classifier, **base_params):
        super().__init__(dataset, image_modality, feature_extractor, task_classifier, **base_params)

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):
        batch_size = int(phi_s.size()[0])
        kernels = losses.gaussian_kernel(phi_s, phi_t, kernel_mul=self._kernel_mul, kernel_num=self._kernel_num)
        return losses.compute_mmd_loss(kernels, batch_size)


class JANTrainerVideo(BaseMMDLikeVideo):
    """This is an implementation of JAN for video data."""

    def __init__(
            self,
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
            kernels = losses.gaussian_kernel(source, target, kernel_mul=k_mul, kernel_num=k_num, fix_sigma=sigma)
            if joint_kernels is not None:
                joint_kernels = joint_kernels * kernels
            else:
                joint_kernels = kernels

        return losses.compute_mmd_loss(joint_kernels, batch_size)


class DANNTrainerVideo(BaseAdaptTrainerVideo, DANNtrainer):
    """This is an implementation of DANN for video data."""

    def __init__(
            self,
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            critic,
            method,
            input_type,
            class_type,
            **base_params,
    ):
        super(DANNTrainerVideo, self).__init__(
            dataset, feature_extractor, task_classifier, critic, method, **base_params
        )
        self.image_modality = image_modality
        self.rgb, self.flow, self.audio = get_image_modality(self.image_modality)
        self.class_type = class_type
        self.verb, self.noun = get_class_type(self.class_type)
        self.rgb_feat = self.feat["rgb"]
        self.flow_feat = self.feat["flow"]
        self.audio_feat = self.feat["audio"]
        self.input_type = input_type

        # Uncomment to store output for EPIC UDA 2021 challenge.(1/3)
        # self.y_hat = []
        # self.y_hat_noun = []
        # self.y_t_hat = []
        # self.y_t_hat_noun = []
        # self.s_id = []
        # self.tu_id = []

    def forward(self, x):
        if self.feat is not None:
            x_rgb = x_flow = x_audio = None
            adversarial_output_rgb = adversarial_output_flow = adversarial_output_audio = None

            # For joint input, both two ifs are used
            if self.rgb:
                x_rgb = self.rgb_feat(x["rgb"])
                x_rgb = x_rgb.view(x_rgb.size(0), -1)
                reverse_feature_rgb = GradReverse.apply(x_rgb, self.alpha)
                adversarial_output_rgb = self.domain_classifier(reverse_feature_rgb)
            if self.flow:
                x_flow = self.flow_feat(x["flow"])
                x_flow = x_flow.view(x_flow.size(0), -1)
                reverse_feature_flow = GradReverse.apply(x_flow, self.alpha)
                adversarial_output_flow = self.domain_classifier(reverse_feature_flow)
            if self.audio:
                x_audio = self.audio_feat(x["audio"])
                x_audio = x_audio.view(x_audio.size(0), -1)
                reverse_feature_audio = GradReverse.apply(x_audio, self.alpha)
                adversarial_output_audio = self.domain_classifier(reverse_feature_audio)

            x = self.concatenate_feature(x_rgb, x_flow, x_audio)

            class_output = self.classifier(x)

            return (
                [x_rgb, x_flow, x_audio],
                class_output,
                [adversarial_output_rgb, adversarial_output_flow, adversarial_output_audio],
            )

    def compute_loss(self, batch, split_name="V"):
        # _s refers to source, _tu refers to unlabeled target
        (
            x_s_rgb,
            x_tu_rgb,
            x_s_flow,
            x_tu_flow,
            x_s_audio,
            x_tu_audio,
            y_s,
            y_tu,
            s_id,
            tu_id,
        ) = self.get_inputs_from_batch(batch)

        _, y_hat, [d_hat_rgb, d_hat_flow, d_hat_audio] = self.forward(
            {"rgb": x_s_rgb, "flow": x_s_flow, "audio": x_s_audio}
        )
        _, y_t_hat, [d_t_hat_rgb, d_t_hat_flow, d_t_hat_audio] = self.forward(
            {"rgb": x_tu_rgb, "flow": x_tu_flow, "audio": x_tu_audio}
        )
        source_batch_size = len(y_s[0])
        target_batch_size = len(y_tu[0])

        if self.rgb:
            loss_dmn_src_rgb, dok_src_rgb = losses.cross_entropy_logits(d_hat_rgb, torch.zeros(source_batch_size))
            loss_dmn_tgt_rgb, dok_tgt_rgb = losses.cross_entropy_logits(d_t_hat_rgb, torch.ones(target_batch_size))
        if self.flow:
            loss_dmn_src_flow, dok_src_flow = losses.cross_entropy_logits(d_hat_flow, torch.zeros(source_batch_size))
            loss_dmn_tgt_flow, dok_tgt_flow = losses.cross_entropy_logits(d_t_hat_flow, torch.ones(target_batch_size))
        if self.audio:
            loss_dmn_src_audio, dok_src_audio = losses.cross_entropy_logits(d_hat_audio, torch.zeros(source_batch_size))
            loss_dmn_tgt_audio, dok_tgt_audio = losses.cross_entropy_logits(
                d_t_hat_audio, torch.ones(target_batch_size)
            )

        # ok is abbreviation for (all) correct, dok refers to domain correct
        if self.rgb:
            if self.flow:
                if self.audio:  # For all inputs
                    loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_flow + loss_dmn_src_audio
                    loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_flow + loss_dmn_tgt_audio
                    dok = torch.cat(
                        (dok_src_rgb, dok_src_flow, dok_src_audio, dok_tgt_rgb, dok_tgt_flow, dok_tgt_audio)
                    )
                    dok_src = torch.cat((dok_src_rgb, dok_src_flow, dok_src_audio))
                    dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_flow, dok_tgt_audio))
                else:  # For joint(rgb+flow) input
                    loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_flow
                    loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_flow
                    dok = torch.cat((dok_src_rgb, dok_src_flow, dok_tgt_rgb, dok_tgt_flow))
                    dok_src = torch.cat((dok_src_rgb, dok_src_flow))
                    dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_flow))
            else:
                if self.audio:  # For rgb+audio input
                    loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_audio
                    loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_audio
                    dok = torch.cat((dok_src_rgb, dok_src_audio, dok_tgt_rgb, dok_tgt_audio))
                    dok_src = torch.cat((dok_src_rgb, dok_src_audio))
                    dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_audio))
                else:  # For rgb input
                    loss_dmn_src = loss_dmn_src_rgb
                    loss_dmn_tgt = loss_dmn_tgt_rgb
                    dok = torch.cat((dok_src_rgb, dok_tgt_rgb))
                    dok_src = dok_src_rgb
                    dok_tgt = dok_tgt_rgb
        else:
            if self.flow:
                if self.audio:  # For flow+audio input
                    loss_dmn_src = loss_dmn_src_flow + loss_dmn_src_audio
                    loss_dmn_tgt = loss_dmn_tgt_flow + loss_dmn_tgt_audio
                    dok = torch.cat((dok_src_flow, dok_src_audio, dok_tgt_flow, dok_tgt_audio))
                    dok_src = torch.cat((dok_src_flow, dok_src_audio))
                    dok_tgt = torch.cat((dok_tgt_flow, dok_tgt_audio))
                else:  # For flow input
                    loss_dmn_src = loss_dmn_src_flow
                    loss_dmn_tgt = loss_dmn_tgt_flow
                    dok = torch.cat((dok_src_flow, dok_tgt_flow))
                    dok_src = dok_src_flow
                    dok_tgt = dok_tgt_flow
            else:  # For audio input
                loss_dmn_src = loss_dmn_src_audio
                loss_dmn_tgt = loss_dmn_tgt_audio
                dok = torch.cat((dok_src_audio, dok_tgt_audio))
                dok_src = dok_src_audio
                dok_tgt = dok_tgt_audio

        task_loss, log_metrics = self.get_loss_log_metrics(split_name, y_hat, y_t_hat, y_s, y_tu, dok)
        adv_loss = loss_dmn_src + loss_dmn_tgt  # adv_loss = src + tgt
        log_metrics.update({f"{split_name}_source_domain_acc": dok_src, f"{split_name}_target_domain_acc": dok_tgt})

        # # Uncomment to store output for EPIC UDA 2021 challenge.(2/3)
        # if split_name == "Te":
        #     self.y_hat.extend(y_hat[0].tolist())
        #     self.y_hat_noun.extend(y_hat[1].tolist())
        #     self.y_t_hat.extend(y_t_hat[0].tolist())
        #     self.y_t_hat_noun.extend(y_t_hat[1].tolist())
        #     self.s_id.extend(s_id)
        #     self.tu_id.extend(tu_id)

        return task_loss, adv_loss, log_metrics


class CDANTrainerVideo(BaseAdaptTrainerVideo, CDANtrainer):
    """This is an implementation of CDAN for video data."""

    def __init__(
            self,
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            critic,
            input_type,
            class_type,
            use_entropy=False,
            use_random=False,
            random_dim=1024,
            **base_params,
    ):
        super(CDANTrainerVideo, self).__init__(
            dataset, feature_extractor, task_classifier, critic, use_entropy, use_random, random_dim, **base_params
        )
        self.image_modality = image_modality
        self.rgb, self.flow, self.audio = get_image_modality(self.image_modality)
        self.class_type = class_type
        self.verb, self.noun = get_class_type(self.class_type)
        self.rgb_feat = self.feat["rgb"]
        self.flow_feat = self.feat["flow"]
        self.audio_feat = self.feat["audio"]
        self.input_type = input_type

    def forward(self, x):
        if self.feat is not None:
            x_rgb = x_flow = x_audio = None
            adversarial_output_rgb = adversarial_output_flow = adversarial_output_audio = None

            # For joint input, both two ifs are used
            if self.rgb:
                x_rgb = self.rgb_feat(x["rgb"])
                x_rgb = x_rgb.view(x_rgb.size(0), -1)
                reverse_feature_rgb = GradReverse.apply(x_rgb, self.alpha)
            if self.flow:
                x_flow = self.flow_feat(x["flow"])
                x_flow = x_flow.view(x_flow.size(0), -1)
                reverse_feature_flow = GradReverse.apply(x_flow, self.alpha)
            if self.audio:
                x_audio = self.audio_feat(x["audio"])
                x_audio = x_audio.view(x_audio.size(0), -1)
                reverse_feature_audio = GradReverse.apply(x_audio, self.alpha)

            x = self.concatenate_feature(x_rgb, x_flow, x_audio)

            class_output = self.classifier(x)
            # # Only use verb class to get softmax_output
            softmax_output = torch.nn.Softmax(dim=1)(class_output[0])
            reverse_out = GradReverse.apply(softmax_output, self.alpha)

            if self.rgb:
                feature_rgb = torch.bmm(reverse_out.unsqueeze(2), reverse_feature_rgb.unsqueeze(1))
                feature_rgb = feature_rgb.view(-1, reverse_out.size(1) * reverse_feature_rgb.size(1))
                if self.random_layer:
                    random_out_rgb = self.random_layer.forward(feature_rgb)
                    adversarial_output_rgb = self.domain_classifier(random_out_rgb.view(-1, random_out_rgb.size(1)))
                else:
                    adversarial_output_rgb = self.domain_classifier(feature_rgb)

            if self.flow:
                feature_flow = torch.bmm(reverse_out.unsqueeze(2), reverse_feature_flow.unsqueeze(1))
                feature_flow = feature_flow.view(-1, reverse_out.size(1) * reverse_feature_flow.size(1))
                if self.random_layer:
                    random_out_flow = self.random_layer.forward(feature_flow)
                    adversarial_output_flow = self.domain_classifier(random_out_flow.view(-1, random_out_flow.size(1)))
                else:
                    adversarial_output_flow = self.domain_classifier(feature_flow)

            if self.audio:
                feature_audio = torch.bmm(reverse_out.unsqueeze(2), reverse_feature_audio.unsqueeze(1))
                feature_audio = feature_audio.view(-1, reverse_out.size(1) * reverse_feature_audio.size(1))
                if self.random_layer:
                    random_out_audio = self.random_layer.forward(feature_audio)
                    adversarial_output_audio = self.domain_classifier(
                        random_out_audio.view(-1, random_out_audio.size(1))
                    )
                else:
                    adversarial_output_audio = self.domain_classifier(feature_audio)

            return (
                [x_rgb, x_flow, x_audio],
                class_output,
                [adversarial_output_rgb, adversarial_output_flow, adversarial_output_audio],
            )

    def compute_loss(self, batch, split_name="V"):
        # _s refers to source, _tu refers to unlabeled target
        (
            x_s_rgb,
            x_tu_rgb,
            x_s_flow,
            x_tu_flow,
            x_s_audio,
            x_tu_audio,
            y_s,
            y_tu,
            s_id,
            tu_id,
        ) = self.get_inputs_from_batch(batch)

        _, y_hat, [d_hat_rgb, d_hat_flow, d_hat_audio] = self.forward(
            {"rgb": x_s_rgb, "flow": x_s_flow, "audio": x_s_audio}
        )
        _, y_t_hat, [d_t_hat_rgb, d_t_hat_flow, d_t_hat_audio] = self.forward(
            {"rgb": x_tu_rgb, "flow": x_tu_flow, "audio": x_tu_audio}
        )
        source_batch_size = len(y_s[0])
        target_batch_size = len(y_tu[0])

        # # Only use verb class to get entropy weights
        if self.entropy:
            e_s = self._compute_entropy_weights(y_hat[0])
            e_t = self._compute_entropy_weights(y_t_hat[0])
            source_weight = e_s / torch.sum(e_s)
            target_weight = e_t / torch.sum(e_t)
        else:
            source_weight = None
            target_weight = None

        if self.rgb:
            loss_dmn_src_rgb, dok_src_rgb = losses.cross_entropy_logits(
                d_hat_rgb, torch.zeros(source_batch_size), source_weight
            )
            loss_dmn_tgt_rgb, dok_tgt_rgb = losses.cross_entropy_logits(
                d_t_hat_rgb, torch.ones(target_batch_size), target_weight
            )

        if self.flow:
            loss_dmn_src_flow, dok_src_flow = losses.cross_entropy_logits(
                d_hat_flow, torch.zeros(source_batch_size), source_weight
            )
            loss_dmn_tgt_flow, dok_tgt_flow = losses.cross_entropy_logits(
                d_t_hat_flow, torch.ones(target_batch_size), target_weight
            )

        if self.audio:
            loss_dmn_src_audio, dok_src_audio = losses.cross_entropy_logits(
                d_hat_audio, torch.zeros(source_batch_size), source_weight
            )
            loss_dmn_tgt_audio, dok_tgt_audio = losses.cross_entropy_logits(
                d_t_hat_audio, torch.ones(target_batch_size), target_weight
            )

        # ok is abbreviation for (all) correct, dok refers to domain correct
        if self.rgb:
            if self.flow:
                if self.audio:  # For all inputs
                    loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_flow + loss_dmn_src_audio
                    loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_flow + loss_dmn_tgt_audio
                    dok = torch.cat(
                        (dok_src_rgb, dok_src_flow, dok_src_audio, dok_tgt_rgb, dok_tgt_flow, dok_tgt_audio)
                    )
                    dok_src = torch.cat((dok_src_rgb, dok_src_flow, dok_src_audio))
                    dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_flow, dok_tgt_audio))
                else:  # For joint(rgb+flow) input
                    loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_flow
                    loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_flow
                    dok = torch.cat((dok_src_rgb, dok_src_flow, dok_tgt_rgb, dok_tgt_flow))
                    dok_src = torch.cat((dok_src_rgb, dok_src_flow))
                    dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_flow))
            else:
                if self.audio:  # For rgb+audio input
                    loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_audio
                    loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_audio
                    dok = torch.cat((dok_src_rgb, dok_src_audio, dok_tgt_rgb, dok_tgt_audio))
                    dok_src = torch.cat((dok_src_rgb, dok_src_audio))
                    dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_audio))
                else:  # For rgb input
                    loss_dmn_src = loss_dmn_src_rgb
                    loss_dmn_tgt = loss_dmn_tgt_rgb
                    dok = torch.cat((dok_src_rgb, dok_tgt_rgb))
                    dok_src = dok_src_rgb
                    dok_tgt = dok_tgt_rgb
        else:
            if self.flow:
                if self.audio:  # For flow+audio input
                    loss_dmn_src = loss_dmn_src_flow + loss_dmn_src_audio
                    loss_dmn_tgt = loss_dmn_tgt_flow + loss_dmn_tgt_audio
                    dok = torch.cat((dok_src_flow, dok_src_audio, dok_tgt_flow, dok_tgt_audio))
                    dok_src = torch.cat((dok_src_flow, dok_src_audio))
                    dok_tgt = torch.cat((dok_tgt_flow, dok_tgt_audio))
                else:  # For flow input
                    loss_dmn_src = loss_dmn_src_flow
                    loss_dmn_tgt = loss_dmn_tgt_flow
                    dok = torch.cat((dok_src_flow, dok_tgt_flow))
                    dok_src = dok_src_flow
                    dok_tgt = dok_tgt_flow
            else:  # For audio input
                loss_dmn_src = loss_dmn_src_audio
                loss_dmn_tgt = loss_dmn_tgt_audio
                dok = torch.cat((dok_src_audio, dok_tgt_audio))
                dok_src = dok_src_audio
                dok_tgt = dok_tgt_audio

        task_loss, log_metrics = self.get_loss_log_metrics(split_name, y_hat, y_t_hat, y_s, y_tu, dok)
        adv_loss = loss_dmn_src + loss_dmn_tgt  # adv_loss = src + tgt
        log_metrics.update({f"{split_name}_source_domain_acc": dok_src, f"{split_name}_target_domain_acc": dok_tgt})

        return task_loss, adv_loss, log_metrics


class WDGRLTrainerVideo(WDGRLtrainer):
    """This is an implementation of WDGRL for video data."""

    def __init__(
            self,
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            critic,
            k_critic=5,
            gamma=10,
            beta_ratio=0,
            **base_params,
    ):
        super(WDGRLTrainerVideo, self).__init__(
            dataset, feature_extractor, task_classifier, critic, k_critic, gamma, beta_ratio, **base_params
        )
        self.image_modality = image_modality
        self.rgb, self.flow, self.audio = get_image_modality(self.image_modality)
        self.rgb_feat = self.feat["rgb"]
        self.flow_feat = self.feat["flow"]

    def forward(self, x):
        if self.feat is not None:
            x_rgb = x_flow = adversarial_output_rgb = adversarial_output_flow = None

            # For joint input, both two ifs are used
            if self.rgb:
                x_rgb = self.rgb_feat(x["rgb"])
                x_rgb = x_rgb.view(x_rgb.size(0), -1)
                adversarial_output_rgb = self.domain_classifier(x_rgb)
            if self.flow:
                x_flow = self.flow_feat(x["flow"])
                x_flow = x_flow.view(x_flow.size(0), -1)
                adversarial_output_flow = self.domain_classifier(x_flow)

            if self.rgb:
                if self.flow:  # For joint input
                    x = torch.cat((x_rgb, x_flow), dim=1)
                else:  # For rgb input
                    x = x_rgb
            else:  # For flow input
                x = x_flow
            class_output = self.classifier(x)

            return [x_rgb, x_flow], class_output, [adversarial_output_rgb, adversarial_output_flow]

    def compute_loss(self, batch, split_name="V"):
        # _s refers to source, _tu refers to unlabeled target
        x_s_rgb = x_tu_rgb = x_s_flow = x_tu_flow = None
        if self.rgb:
            if self.flow:  # For joint input
                (x_s_rgb, y_s), (x_s_flow, y_s_flow), (x_tu_rgb, y_tu), (x_tu_flow, y_tu_flow) = batch
            else:  # For rgb input
                (x_s_rgb, y_s), (x_tu_rgb, y_tu) = batch
        else:  # For flow input
            (x_s_flow, y_s), (x_tu_flow, y_tu) = batch

        _, y_hat, [d_hat_rgb, d_hat_flow] = self.forward({"rgb": x_s_rgb, "flow": x_s_flow})
        _, y_t_hat, [d_t_hat_rgb, d_t_hat_flow] = self.forward({"rgb": x_tu_rgb, "flow": x_tu_flow})
        batch_size = len(y_s)

        # ok is abbreviation for (all) correct, dok refers to domain correct
        if self.rgb:
            _, dok_src_rgb = losses.cross_entropy_logits(d_hat_rgb, torch.zeros(batch_size))
            _, dok_tgt_rgb = losses.cross_entropy_logits(d_t_hat_rgb, torch.ones(batch_size))
        if self.flow:
            _, dok_src_flow = losses.cross_entropy_logits(d_hat_flow, torch.zeros(batch_size))
            _, dok_tgt_flow = losses.cross_entropy_logits(d_t_hat_flow, torch.ones(batch_size))

        if self.rgb and self.flow:  # For joint input
            dok = torch.cat((dok_src_rgb, dok_src_flow, dok_tgt_rgb, dok_tgt_flow))
            dok_src = torch.cat((dok_src_rgb, dok_src_flow))
            dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_flow))
            wasserstein_distance_rgb = d_hat_rgb.mean() - (1 + self._beta_ratio) * d_t_hat_rgb.mean()
            wasserstein_distance_flow = d_hat_flow.mean() - (1 + self._beta_ratio) * d_t_hat_flow.mean()
            wasserstein_distance = (wasserstein_distance_rgb + wasserstein_distance_flow) / 2
        else:
            if self.rgb:  # For rgb input
                d_hat = d_hat_rgb
                d_t_hat = d_t_hat_rgb
                dok_src = dok_src_rgb
                dok_tgt = dok_tgt_rgb
            else:  # For flow input
                d_hat = d_hat_flow
                d_t_hat = d_t_hat_flow
                dok_src = dok_src_flow
                dok_tgt = dok_tgt_flow

            wasserstein_distance = d_hat.mean() - (1 + self._beta_ratio) * d_t_hat.mean()
            dok = torch.cat((dok_src, dok_tgt))

        loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)
        adv_loss = wasserstein_distance
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": dok,
            f"{split_name}_source_domain_acc": dok_src,
            f"{split_name}_target_domain_acc": dok_tgt,
            f"{split_name}_wasserstein_dist": wasserstein_distance,
        }
        return task_loss, adv_loss, log_metrics

    def configure_optimizers(self):
        if self.image_modality in ["rgb", "flow"]:
            if self.rgb_feat is not None:
                nets = [self.rgb_feat, self.classifier]
            else:
                nets = [self.flow_feat, self.classifier]
        elif self.image_modality == "joint":
            nets = [self.rgb_feat, self.flow_feat, self.classifier]
        parameters = set()

        for net in nets:
            parameters |= set(net.parameters())

        if self._adapt_lr:
            task_feat_optimizer, task_feat_sched = self._configure_optimizer(parameters)
            self.critic_opt, self.critic_sched = self._configure_optimizer(self.domain_classifier.parameters())
            self.critic_opt = self.critic_opt[0]
            self.critic_sched = self.critic_sched[0]
            return task_feat_optimizer, task_feat_sched
        else:
            task_feat_optimizer = self._configure_optimizer(parameters)
            self.critic_opt = self._configure_optimizer(self.domain_classifier.parameters())
            self.critic_sched = None
            self.critic_opt = self.critic_opt[0]
        return task_feat_optimizer

    def critic_update_steps(self, batch):
        if self.current_epoch < self._init_epochs:
            return

        set_requires_grad(self.domain_classifier, requires_grad=True)

        if self.image_modality in ["rgb", "flow"]:
            if self.rgb_feat is not None:
                set_requires_grad(self.rgb_feat, requires_grad=False)
                (x_s, y_s), (x_tu, _) = batch
                with torch.no_grad():
                    h_s = self.rgb_feat(x_s).data.view(x_s.shape[0], -1)
                    h_t = self.rgb_feat(x_tu).data.view(x_tu.shape[0], -1)
            else:
                set_requires_grad(self.flow_feat, requires_grad=False)
                (x_s, y_s), (x_tu, _) = batch
                with torch.no_grad():
                    h_s = self.flow_feat(x_s).data.view(x_s.shape[0], -1)
                    h_t = self.flow_feat(x_tu).data.view(x_tu.shape[0], -1)

            for _ in range(self._k_critic):
                # gp refers to gradient penelty in Wasserstein distance.
                gp = losses.gradient_penalty(self.domain_classifier, h_s, h_t)

                critic_s = self.domain_classifier(h_s)
                critic_t = self.domain_classifier(h_t)
                wasserstein_distance = critic_s.mean() - (1 + self._beta_ratio) * critic_t.mean()

                critic_cost = -wasserstein_distance + self._gamma * gp

                self.critic_opt.zero_grad()
                critic_cost.backward()
                self.critic_opt.step()
                if self.critic_sched:
                    self.critic_sched.step()

            if self.rgb_feat is not None:
                set_requires_grad(self.rgb_feat, requires_grad=True)
            else:
                set_requires_grad(self.flow_feat, requires_grad=True)
            set_requires_grad(self.domain_classifier, requires_grad=False)

        elif self.image_modality == "joint":
            set_requires_grad(self.rgb_feat, requires_grad=False)
            set_requires_grad(self.flow_feat, requires_grad=False)
            (x_s_rgb, y_s), (x_s_flow, _), (x_tu_rgb, _), (x_tu_flow, _) = batch
            with torch.no_grad():
                h_s_rgb = self.rgb_feat(x_s_rgb).data.view(x_s_rgb.shape[0], -1)
                h_t_rgb = self.rgb_feat(x_tu_rgb).data.view(x_tu_rgb.shape[0], -1)
                h_s_flow = self.flow_feat(x_s_flow).data.view(x_s_flow.shape[0], -1)
                h_t_flow = self.flow_feat(x_tu_flow).data.view(x_tu_flow.shape[0], -1)
                h_s = torch.cat((h_s_rgb, h_s_flow), dim=1)
                h_t = torch.cat((h_t_rgb, h_t_flow), dim=1)

            # Need to improve to process rgb and flow separately in the future.
            for _ in range(self._k_critic):
                # gp_x refers to gradient penelty for the input with the image_modality x.
                gp_rgb = losses.gradient_penalty(self.domain_classifier, h_s_rgb, h_t_rgb)
                gp_flow = losses.gradient_penalty(self.domain_classifier, h_s_flow, h_t_flow)

                critic_s_rgb = self.domain_classifier(h_s_rgb)
                critic_s_flow = self.domain_classifier(h_s_flow)
                critic_t_rgb = self.domain_classifier(h_t_rgb)
                critic_t_flow = self.domain_classifier(h_t_flow)
                wasserstein_distance_rgb = critic_s_rgb.mean() - (1 + self._beta_ratio) * critic_t_rgb.mean()
                wasserstein_distance_flow = critic_s_flow.mean() - (1 + self._beta_ratio) * critic_t_flow.mean()

                critic_cost = (
                                      -wasserstein_distance_rgb
                                      + -wasserstein_distance_flow
                                      + self._gamma * gp_rgb
                                      + self._gamma * gp_flow
                              ) * 0.5

                self.critic_opt.zero_grad()
                critic_cost.backward()
                self.critic_opt.step()
                if self.critic_sched:
                    self.critic_sched.step()

            set_requires_grad(self.rgb_feat, requires_grad=True)
            set_requires_grad(self.flow_feat, requires_grad=True)
            set_requires_grad(self.domain_classifier, requires_grad=False)


# class TA3NtrainerVideoTemp(BaseAdaptTrainerVideo):
#     """This is an implementation of TA3N for video data."""
#
#     def __init__(
#         self,
#         dataset,
#         image_modality,
#         feature_extractor,
#         task_classifier,
#         critic,
#         method,
#         input_type,
#         class_type,
#         **base_params,
#     ):
#         super(TA3NtrainerVideoTemp, self).__init__(
#             dataset,
#             image_modality,
#             feature_extractor,
#             task_classifier,
#             critic,
#             method,
#             input_type,
#             class_type,
#             **base_params,
#         )
#         # self.image_modality = image_modality
#         # self.rgb, self.flow, self.audio = get_image_modality(self.image_modality)
#         # self.class_type = class_type
#         # self.verb, self.noun = get_class_type(self.class_type)
#         # if self.method is not Method.TA3N:
#         #     self.rgb_feat = self.feat["rgb"]
#         #     self.flow_feat = self.feat["flow"]
#         #     self.audio_feat = self.feat["audio"]
#         # else:
#         #     self.feat = self.feat["all"]
#         # self.input_type = input_type
#
#         self.feat = self.feat["all"]
#         self.correct_batch_size_source = None
#         self.correct_batch_size_target = None
#
#         self._init_epochs = 0
#         self.batch_size = [128, 128, 128]
#         self.dann_warmup = True
#         self.beta = [0.75, 0.75, 0.5]
#
#         self.attention = TemporalAttention()
#
#         # for saving the details of the individual layers
#         f = open("modules_list.txt", "w")
#         f.write("Pykale version!\n")
#         module_list = []
#         for module in self.named_children():
#             for m in module[1].named_children():
#                 module_list.append(str(m[1]) + "\t" + str(m[0]) + " | inside " + str(module[0]))
#
#         for m in sorted(module_list):
#             f.write(str(m) + "\n")
#         f.close()
#
#     def forward(self, x):
#         if self.feat is not None:
#             x_rgb = x_flow = x_audio = None
#             adversarial_output_rgb = adversarial_output_flow = adversarial_output_audio = None
#             adv_output_rgb_1 = adv_output_flow_1 = adv_output_audio_1 = None
#             adv_output_rgb_0 = adv_output_flow_0 = adv_output_audio_0 = None
#
#             # For joint input, both two ifs are used
#             if self.method is not Method.TA3N:
#
#                 if self.rgb:
#                     x_rgb = self.rgb_feat(x["rgb"])
#                     if self.method is Method.TA3N:
#                         x_rgb, adv_output_rgb_0, adv_output_rgb_1 = self.rgb_attention(x_rgb, self.beta)
#                     x_rgb = x_rgb.view(x_rgb.size(0), -1)
#                     reverse_feature_rgb = GradReverse.apply(x_rgb, self.beta[1])
#                     adversarial_output_rgb = self.domain_classifier(reverse_feature_rgb)
#                 if self.flow:
#                     x_flow = self.flow_feat(x["flow"])
#                     if self.method is Method.TA3N:
#                         x_flow, adv_output_flow_0, adv_output_flow_1 = self.flow_attention(x_flow, self.beta)
#                     x_flow = x_flow.view(x_flow.size(0), -1)
#                     reverse_feature_flow = GradReverse.apply(x_flow, self.beta[1])
#                     adversarial_output_flow = self.domain_classifier(reverse_feature_flow)
#                 if self.audio:
#                     x_audio = self.audio_feat(x["audio"])
#                     if self.method is Method.TA3N:
#                         x_audio, adv_output_audio_0, adv_output_audio_1 = self.audio_attention(x_audio, self.beta)
#                     x_audio = x_audio.view(x_audio.size(0), -1)
#                     reverse_feature_audio = GradReverse.apply(x_audio, self.beta[1])
#                     adversarial_output_audio = self.domain_classifier(reverse_feature_audio)
#
#                 x = self.concatenate_feature(x_rgb, x_flow, x_audio)
#
#             else:
#                 x = self.concatenate_feature(x["rgb"], x["flow"], x["audio"])
#                 x = self.feat(x)
#                 x, adv_output_0, adv_output_1 = self.attention(x, self.beta)
#                 x = x.view(x.size(0), -1)
#                 reverse_feature = GradReverse.apply(x, self.beta[1])
#                 adversarial_output = self.domain_classifier(reverse_feature)
#                 adversarial_output_rgb = adversarial_output_flow = adversarial_output_audio = adversarial_output
#                 x_rgb = x_flow = x_audio = x
#                 adv_output_rgb_0 = adv_output_flow_0 = adv_output_audio_0 = adv_output_0
#                 adv_output_rgb_1 = adv_output_flow_1 = adv_output_audio_1 = adv_output_1
#
#             class_output = self.classifier(x)
#
#             return (
#                 [x_rgb, x_flow, x_audio],
#                 class_output,
#                 [adversarial_output_rgb, adversarial_output_flow, adversarial_output_audio],
#                 [
#                     [adv_output_rgb_0, adv_output_rgb_1],
#                     [adv_output_flow_0, adv_output_flow_1],
#                     [adv_output_audio_0, adv_output_audio_1],
#                 ],
#             )
#
#     def compute_loss(self, batch, split_name="V"):
#         # _s refers to source, _tu refers to unlabeled target
#         (
#             x_s_rgb,
#             x_tu_rgb,
#             x_s_flow,
#             x_tu_flow,
#             x_s_audio,
#             x_tu_audio,
#             y_s,
#             y_tu,
#             s_id,
#             tu_id,
#         ) = self.get_inputs_from_batch(batch)
#
#         source_batch_size = len(y_s[0])
#         target_batch_size = len(y_tu[0])
#
#         if self.correct_batch_size_source is None or self.correct_batch_size_target is None:
#             self.correct_batch_size_source = source_batch_size
#             self.correct_batch_size_target = target_batch_size
#         else:
#             if source_batch_size != self.correct_batch_size_source:
#                 x_s_rgb, x_s_flow, x_s_audio = self.add_dummy_data(
#                     x_s_rgb, x_s_flow, x_s_audio, self.correct_batch_size_source
#                 )
#             if target_batch_size != self.correct_batch_size_target:
#                 x_tu_rgb, x_tu_flow, x_tu_audio = self.add_dummy_data(
#                     x_tu_rgb, x_tu_flow, x_tu_audio, self.correct_batch_size_target
#                 )
#
#         _, y_hat, [d_hat_rgb, d_hat_flow, d_hat_audio], d_hat_0_1 = self.forward(
#             {"rgb": x_s_rgb, "flow": x_s_flow, "audio": x_s_audio}
#         )
#         _, y_t_hat, [d_t_hat_rgb, d_t_hat_flow, d_t_hat_audio], d_t_hat_0_1 = self.forward(
#             {"rgb": x_tu_rgb, "flow": x_tu_flow, "audio": x_tu_audio}
#         )
#
#         if source_batch_size != self.correct_batch_size_source:
#             y_hat, d_hat_rgb, d_hat_flow, d_hat_audio, d_hat_0_1 = self.remove_dummy(
#                 y_hat, d_hat_rgb, d_hat_flow, d_hat_audio, d_hat_0_1, source_batch_size
#             )
#         if target_batch_size != self.correct_batch_size_target:
#             y_t_hat, d_t_hat_rgb, d_t_hat_flow, d_t_hat_audio, d_t_hat_0_1 = self.remove_dummy(
#                 y_t_hat, d_t_hat_rgb, d_t_hat_flow, d_t_hat_audio, d_t_hat_0_1, target_batch_size
#             )
#
#         if self.rgb:
#             loss_dmn_src_rgb, dok_src_rgb = losses.cross_entropy_logits(d_hat_rgb, torch.zeros(source_batch_size))
#             loss_dmn_tgt_rgb, dok_tgt_rgb = losses.cross_entropy_logits(d_t_hat_rgb, torch.ones(target_batch_size))
#         if self.flow:
#             loss_dmn_src_flow, dok_src_flow = losses.cross_entropy_logits(d_hat_flow, torch.zeros(source_batch_size))
#             loss_dmn_tgt_flow, dok_tgt_flow = losses.cross_entropy_logits(d_t_hat_flow, torch.ones(target_batch_size))
#         if self.audio:
#             loss_dmn_src_audio, dok_src_audio = losses.cross_entropy_logits(d_hat_audio, torch.zeros(source_batch_size))
#             loss_dmn_tgt_audio, dok_tgt_audio = losses.cross_entropy_logits(
#                 d_t_hat_audio, torch.ones(target_batch_size)
#             )
#
#         # ok is abbreviation for (all) correct, dok refers to domain correct
#         if self.rgb:
#             if self.flow:
#                 if self.audio:  # For all inputs
#                     loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_flow + loss_dmn_src_audio
#                     loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_flow + loss_dmn_tgt_audio
#                     dok = torch.cat(
#                         (dok_src_rgb, dok_src_flow, dok_src_audio, dok_tgt_rgb, dok_tgt_flow, dok_tgt_audio)
#                     )
#                     dok_src = torch.cat((dok_src_rgb, dok_src_flow, dok_src_audio))
#                     dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_flow, dok_tgt_audio))
#                 else:  # For joint(rgb+flow) input
#                     loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_flow
#                     loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_flow
#                     dok = torch.cat((dok_src_rgb, dok_src_flow, dok_tgt_rgb, dok_tgt_flow))
#                     dok_src = torch.cat((dok_src_rgb, dok_src_flow))
#                     dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_flow))
#             else:
#                 if self.audio:  # For rgb+audio input
#                     loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_audio
#                     loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_audio
#                     dok = torch.cat((dok_src_rgb, dok_src_audio, dok_tgt_rgb, dok_tgt_audio))
#                     dok_src = torch.cat((dok_src_rgb, dok_src_audio))
#                     dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_audio))
#                 else:  # For rgb input
#                     loss_dmn_src = loss_dmn_src_rgb
#                     loss_dmn_tgt = loss_dmn_tgt_rgb
#                     dok = torch.cat((dok_src_rgb, dok_tgt_rgb))
#                     dok_src = dok_src_rgb
#                     dok_tgt = dok_tgt_rgb
#         else:
#             if self.flow:
#                 if self.audio:  # For flow+audio input
#                     loss_dmn_src = loss_dmn_src_flow + loss_dmn_src_audio
#                     loss_dmn_tgt = loss_dmn_tgt_flow + loss_dmn_tgt_audio
#                     dok = torch.cat((dok_src_flow, dok_src_audio, dok_tgt_flow, dok_tgt_audio))
#                     dok_src = torch.cat((dok_src_flow, dok_src_audio))
#                     dok_tgt = torch.cat((dok_tgt_flow, dok_tgt_audio))
#                 else:  # For flow input
#                     loss_dmn_src = loss_dmn_src_flow
#                     loss_dmn_tgt = loss_dmn_tgt_flow
#                     dok = torch.cat((dok_src_flow, dok_tgt_flow))
#                     dok_src = dok_src_flow
#                     dok_tgt = dok_tgt_flow
#             else:  # For audio input
#                 loss_dmn_src = loss_dmn_src_audio
#                 loss_dmn_tgt = loss_dmn_tgt_audio
#                 dok = torch.cat((dok_src_audio, dok_tgt_audio))
#                 dok_src = dok_src_audio
#                 dok_tgt = dok_tgt_audio
#
#         task_loss, log_metrics = self.get_loss_log_metrics(split_name, y_hat, y_t_hat, y_s, y_tu, dok)
#         adv_loss = loss_dmn_src + loss_dmn_tgt  # adv_loss = src + tgt
#         log_metrics.update({f"{split_name}_source_domain_acc": dok_src, f"{split_name}_target_domain_acc": dok_tgt})
#
#         adv_loss_0_1 = 0
#         for i in range(len(d_hat_0_1)):
#             if d_hat_0_1[i] is not None and d_hat_0_1[i][0] is not None:
#                 preds_s = d_hat_0_1[i]
#                 preds_t = d_t_hat_0_1[i]
#                 for pred_s, pred_t in zip(preds_s, preds_t):
#                     pred_s = pred_s.view(-1, pred_s.size()[-1])
#                     pred_t = pred_t.view(-1, pred_t.size()[-1])
#                     d_label_s = torch.zeros(pred_s.size(0))
#                     d_label_t = torch.ones(pred_t.size(0))
#
#                     pred = torch.cat((pred_s, pred_t), 0)
#                     d_label = torch.cat((d_label_s, d_label_t), 0)
#
#                     adv_loss_0_1 += losses.cross_entropy_logits(pred, d_label.type_as(pred).long())[0]
#         adv_loss += adv_loss_0_1
#
#         if self.method is Method.TA3N:
#             mmd_loss = 0
#             if self.rgb:
#                 losses_mmd = [losses.mmd_rbf(x_s_rgb[t], x_tu_rgb[t]) for t in range(x_s_rgb.size(0))]
#                 mmd_loss += sum(losses_mmd) / len(losses_mmd)
#             if self.flow:
#                 losses_mmd = [losses.mmd_rbf(x_s_flow[t], x_tu_flow[t]) for t in range(x_s_flow.size(0))]
#                 mmd_loss += sum(losses_mmd) / len(losses_mmd)
#             if self.audio:
#                 losses_mmd = [losses.mmd_rbf(x_s_audio[t], x_tu_audio[t]) for t in range(x_s_audio.size(0))]
#                 mmd_loss += sum(losses_mmd) / len(losses_mmd)
#             self.alpha_mmd = 0
#             task_loss += self.alpha_mmd * mmd_loss
#             # task_loss += loss_attn
#         # entropy loss for target data
#         if self.method is Method.TA3N:
#             self.gamma = 0.003
#             if self.verb:
#                 loss_entropy_verb = losses.cross_entropy_soft(y_t_hat[0])
#             if self.noun:
#                 loss_entropy_noun = losses.cross_entropy_soft(y_t_hat[1])
#             if self.verb and not self.noun:
#                 task_loss += self.gamma * loss_entropy_verb
#             elif self.verb and self.noun:
#                 task_loss += self.gamma * 0.5 * (loss_entropy_verb + loss_entropy_noun)
#
#         # attentive entropy loss
#         if self.method is Method.TA3N:
#             if self.verb:
#                 loss_entropy_verb = 0
#                 if self.rgb:
#                     loss_entropy_verb += losses.attentive_entropy(
#                         torch.cat((y_hat[0], y_t_hat[0]), 0), torch.cat((d_hat_rgb, d_t_hat_rgb), 0)
#                     )
#                 if self.flow:
#                     loss_entropy_verb += losses.attentive_entropy(
#                         torch.cat((y_hat[0], y_t_hat[0]), 0), torch.cat((d_hat_flow, d_t_hat_flow), 0)
#                     )
#                 if self.audio:
#                     loss_entropy_verb += losses.attentive_entropy(
#                         torch.cat((y_hat[0], y_t_hat[0]), 0), torch.cat((d_hat_audio, d_t_hat_audio), 0)
#                     )
#             if self.noun:
#                 loss_entropy_noun = 0
#                 if self.rgb:
#                     loss_entropy_noun += losses.attentive_entropy(
#                         torch.cat((y_hat[1], y_t_hat[1]), 0), torch.cat((d_hat_rgb, d_t_hat_rgb), 0)
#                     )
#                 if self.flow:
#                     loss_entropy_noun += losses.attentive_entropy(
#                         torch.cat((y_hat[1], y_t_hat[1]), 0), torch.cat((d_hat_flow, d_t_hat_flow), 0)
#                     )
#                 if self.audio:
#                     loss_entropy_noun += losses.attentive_entropy(
#                         torch.cat((y_hat[1], y_t_hat[1]), 0), torch.cat((d_hat_audio, d_t_hat_audio), 0)
#                     )
#
#             if self.verb and not self.noun:
#                 task_loss += self.gamma * loss_entropy_verb
#             elif self.verb and self.noun:
#                 task_loss += self.gamma * 0.5 * (loss_entropy_verb + loss_entropy_noun)
#
#         # # Uncomment to store output for EPIC UDA 2021 challenge.(2/3)
#         # if split_name == "Te":
#         #     self.y_hat.extend(y_hat[0].tolist())
#         #     self.y_hat_noun.extend(y_hat[1].tolist())
#         #     self.y_t_hat.extend(y_t_hat[0].tolist())
#         #     self.y_t_hat_noun.extend(y_t_hat[1].tolist())
#         #     self.s_id.extend(s_id)
#         #     self.tu_id.extend(tu_id)
#
#         return task_loss, adv_loss, log_metrics
#
#     def add_dummy_data(self, x_rgb, x_flow, x_audio, batch_size):
#         if self.rgb:
#             current_size = x_rgb.size()
#             data_dummy = torch.zeros(batch_size - current_size[0], current_size[1], current_size[2])
#             data_dummy = data_dummy.type_as(x_rgb)
#             x_rgb = torch.cat((x_rgb, data_dummy))
#         if self.flow:
#             current_size = x_flow.size()
#             data_dummy = torch.zeros(batch_size - current_size[0], current_size[1], current_size[2])
#             data_dummy = data_dummy.type_as(x_flow)
#             x_flow = torch.cat((x_flow, data_dummy))
#         if self.audio:
#             current_size = x_audio.size()
#             data_dummy = torch.zeros(batch_size - current_size[0], current_size[1], current_size[2])
#             data_dummy = data_dummy.type_as(x_audio)
#             x_audio = torch.cat((x_audio, data_dummy))
#         return x_rgb, x_flow, x_audio
#
#     def remove_dummy(self, y_hat, d_hat_rgb, d_hat_flow, d_hat_audio, d_hat_0_1, batch_size):
#         y_hat[0] = y_hat[0][:batch_size]
#         y_hat[1] = y_hat[1][:batch_size]
#         if self.rgb:
#             d_hat_rgb = d_hat_rgb[:batch_size]
#         if self.flow:
#             d_hat_flow = d_hat_flow[:batch_size]
#         if self.audio:
#             d_hat_audio = d_hat_audio[:batch_size]
#         d_hat_0_1 = [
#             [d[0][:batch_size], d[1][:batch_size]] if d is not None and d[0] is not None else [None, None]
#             for d in d_hat_0_1
#         ]
#         return y_hat, d_hat_rgb, d_hat_flow, d_hat_audio, d_hat_0_1
#
#     def _update_batch_epoch_factors(self, batch_id):
#         ## setup hyperparameters
#         loss_c_current = 999  # random large number
#         loss_c_previous = 999  # random large number
#
#         start_steps = len(self.trainer.train_dataloader)
#         total_steps = self.epochs * len(self.trainer.train_dataloader)
#         p = float(self.global_step + start_steps) / total_steps
#         self.beta_dann = 2.0 / (1.0 + np.exp(-1.0 * p)) - 1
#         # replace the default beta if value < 0
#         self.beta = [self.beta_dann if self.beta[i] < 0 else self.beta[i] for i in range(len(self.beta))]
#         if self.dann_warmup:
#             self.beta_new = [self.beta_dann * self.beta[i] for i in range(len(self.beta))]
#         else:
#             self.beta_new = self.beta
#
#         # print("i+start_steps: {}, total_steps: {}, p :{}, beta_new: {}".format(float(self.global_step + start_steps), total_steps, p, self.beta_new))
#         # print("lr: {}, alpha: {}, mu: {}".format(self.optimizers().param_groups[0]['lr'], self.alpha, self.mu))
#
#         ## schedule for learning rate
#         if self.lr_adaptive == "loss":
#             self.adjust_learning_rate_loss(self.optimizers(), self.lr_decay, loss_c_current, loss_c_previous, ">")
#         elif self.lr_adaptive is None:
#             if self.global_step in [i * start_steps for i in self.lr_steps]:
#                 self.adjust_learning_rate(self.optimizers(), self.lr_decay)
#
#         if self.lr_adaptive == "dann":
#             self.adjust_learning_rate_dann(self.optimizers(), p)
#
#         self.alpha = 2 / (1 + math.exp(-1 * (self.current_epoch) / self.epochs)) - 1 if self.alpha < 0 else self.alpha
#
#     def training_step(self, batch, batch_nb):
#         """Automatically called by lightning while training a single batch
#
#         Args:
#             batch: an item of the dataloader(s) passed with the trainer
#             batch_nb: the batch index (which batch is currently being trained)
#         Returns:
#             The loss(es) calculated after performing an optimiser step
#         """
#
#         self._update_batch_epoch_factors(batch_nb)
#
#         loss, task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="T")
#
#         log_metrics = get_aggregated_metrics_from_dict(log_metrics)
#         log_metrics.update(get_metrics_from_parameter_dict(self.get_parameters_watch_list(), loss.device))
#         log_metrics["T_total_loss"] = loss
#         log_metrics["T_adv_loss"] = adv_loss
#         log_metrics["T_task_loss"] = task_loss
#
#         for key in log_metrics:
#             self.log(key, log_metrics[key])
#
#         return {"loss": log_metrics["Loss Total"]}


class TA3NTrainerVideo(BaseAdaptTrainerVideo):
    def __init__(
            self,
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            critic,
            method,
            input_type,
            class_type,
            # **base_params,

            dict_n_class,
            init_lr,
            batch_size,
            optimizer,
            baseline_type,
            frame_aggregation,
            alpha,
            beta,
            gamma,
            mu,
            adv_da,
            use_target,
            place_adv,
            pred_normalize,
            add_loss_da,
            nb_adapt_epochs,
            dann_warmup,
            lr_adaptive,
            lr_steps,
            lr_decay,

            num_segments=5,
            # val_segments=25,
            arch="TBN",
            # path_pretrained="",

            new_length=None,
            before_softmax=True,
            dropout_i=0.5,
            dropout_v=0.5,
            use_bn=None,
            ens_DA=None,
            crop_num=1,
            partial_bn=True,
            verbose=True,
            add_fc=1,
            fc_dim=1024,
            n_rnn=1,
            rnn_cell="LSTM",
            n_directions=1,
            n_ts=5,
            use_attn="TransAttn",
            n_attn=1,
            use_attn_frame=None,
            share_params="Y",
    ):
        super(TA3NTrainerVideo, self).__init__(dataset, feature_extractor, task_classifier)

        self.image_modality = image_modality
        self.rgb, self.flow, self.audio = get_image_modality(self.image_modality)
        self.class_type = class_type
        self.verb, self.noun = get_class_type(self.class_type)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_domain = torch.nn.CrossEntropyLoss()

        self.train_metric = 'all'
        self.adv_DA = adv_da
        self.use_target = use_target
        self.place_adv = place_adv
        self.pred_normalize = pred_normalize
        self.add_loss_DA = add_loss_da
        self.labels_available = False
        self.nb_adapt_epochs = nb_adapt_epochs
        self.dann_warmup = dann_warmup
        self.lr_adaptive = lr_adaptive
        self.lr_steps = lr_steps
        self.lr_decay = lr_decay

        self._method = method

        # self._init_lambda = lambda_init
        # self.lamb_da = lambda_init
        # self._adapt_lambda = adapt_lambda
        # self._adapt_lr = adapt_lr

        # self._init_epochs = nb_init_epochs
        # self._non_init_epochs = nb_adapt_epochs - self._init_epochs
        # assert self._non_init_epochs > 0
        self._batch_size = batch_size
        self._init_lr = init_lr
        # self._lr_fact = 1.0
        # self._grow_fact = 0.0
        self._dataset = dataset
        self.feat = feature_extractor
        self.classifier = task_classifier
        self._dataset.prepare_data_loaders()
        # self._nb_training_batches = None  # to be set by method train_dataloader
        self._optimizer_params = optimizer

        self.dict_n_class = dict_n_class
        self.modality = image_modality
        self.train_segments = num_segments
        self.val_segments = num_segments
        self.baseline_type = baseline_type
        self.frame_aggregation = frame_aggregation
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout_rate_i = dropout_i
        self.dropout_rate_v = dropout_v
        self.use_bn = use_bn
        self.ens_DA = ens_DA
        self.crop_num = crop_num
        self.add_fc = add_fc
        self.fc_dim = fc_dim
        self.share_params = share_params

        self.base_model = arch
        self.verbose = verbose
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.gamma = gamma

        # RNN
        self.n_layers = n_rnn
        self.rnn_cell = rnn_cell
        self.n_directions = n_directions
        self.n_ts = n_ts  # temporal segment

        # Attention
        self.use_attn = use_attn
        self.n_attn = n_attn
        self.use_attn_frame = use_attn_frame

        if new_length is None:
            self.new_length = 1 if image_modality == "RGB" else 5
        else:
            self.new_length = new_length

        # if verbose:
        #     log_info(
        #         (
        #             """
        #         Initializing TSN with base model: {}.
        #         TSN Configurations:
        #         input_modality:     {}
        #         num_segments:       {}
        #         new_length:         {}
        #         """.format(
        #                 self.base_model, self.modality, self.train_segments, self.new_length
        #             )
        #         )
        #     )

        self._prepare_DA(self.dict_n_class, self.base_model, self.modality)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

        self.no_partialbn = not partial_bn

        self.best_prec1 = 0

        # ======= Setting up losses and the criterion ======#

        self.loss_c_current = 0
        self.loss_c_previous = 0
        self.save_attention = -1

        self.end = time.time()
        self.end_val = time.time()

        # ======= For lightning to use custom optimizer ======#
        self.automatic_optimization = True

        self.tensorboard = False

    def _prepare_DA(self, num_class, base_model, modality):  # convert the model to DA framework
        if base_model == "TBN" and modality == "ALL":
            self.feature_dim = 3072
        elif base_model == "TBN":
            self.feature_dim = 1024
        else:
            model_test = getattr(torchvision.models, base_model)(True)  # model_test is only used for getting the dim #
            self.feature_dim = model_test.fc.in_features

        std = 0.001
        feat_shared_dim = (
            min(self.fc_dim, self.feature_dim) if self.add_fc > 0 and self.fc_dim > 0 else self.feature_dim
        )
        feat_frame_dim = feat_shared_dim

        self.relu = nn.ReLU(inplace=True)
        self.dropout_i = nn.Dropout(p=self.dropout_rate_i)
        self.dropout_v = nn.Dropout(p=self.dropout_rate_v)

        # ------ frame-level layers (shared layers + source layers + domain layers) ------#
        if self.add_fc < 1:
            raise ValueError("add at least one fc layer")

        # 1. shared feature layers
        self.fc_feature_shared_source = nn.Linear(self.feature_dim, feat_shared_dim)
        normal_(self.fc_feature_shared_source.weight, 0, std)
        constant_(self.fc_feature_shared_source.bias, 0)

        if self.add_fc > 1:
            self.fc_feature_shared_2_source = nn.Linear(feat_shared_dim, feat_shared_dim)
            normal_(self.fc_feature_shared_2_source.weight, 0, std)
            constant_(self.fc_feature_shared_2_source.bias, 0)

        if self.add_fc > 2:
            self.fc_feature_shared_3_source = nn.Linear(feat_shared_dim, feat_shared_dim)
            normal_(self.fc_feature_shared_3_source.weight, 0, std)
            constant_(self.fc_feature_shared_3_source.bias, 0)

        # 2. frame-level feature layers
        self.fc_feature_source = nn.Linear(feat_shared_dim, feat_frame_dim)
        normal_(self.fc_feature_source.weight, 0, std)
        constant_(self.fc_feature_source.bias, 0)

        # 3. domain feature layers (frame-level)
        self.fc_feature_domain = nn.Linear(feat_shared_dim, feat_frame_dim)
        normal_(self.fc_feature_domain.weight, 0, std)
        constant_(self.fc_feature_domain.bias, 0)

        # 4. classifiers (frame-level)
        self.fc_classifier_source_verb = nn.Linear(feat_frame_dim, num_class["verb"])
        self.fc_classifier_source_noun = nn.Linear(feat_frame_dim, num_class["noun"])
        normal_(self.fc_classifier_source_verb.weight, 0, std)
        constant_(self.fc_classifier_source_verb.bias, 0)
        normal_(self.fc_classifier_source_noun.weight, 0, std)
        constant_(self.fc_classifier_source_noun.bias, 0)

        self.fc_classifier_domain = nn.Linear(feat_frame_dim, 2)
        normal_(self.fc_classifier_domain.weight, 0, std)
        constant_(self.fc_classifier_domain.bias, 0)

        if self.share_params == "N":
            self.fc_feature_shared_target = nn.Linear(self.feature_dim, feat_shared_dim)
            normal_(self.fc_feature_shared_target.weight, 0, std)
            constant_(self.fc_feature_shared_target.bias, 0)
            if self.add_fc > 1:
                self.fc_feature_shared_2_target = nn.Linear(feat_shared_dim, feat_shared_dim)
                normal_(self.fc_feature_shared_2_target.weight, 0, std)
                constant_(self.fc_feature_shared_2_target.bias, 0)
            if self.add_fc > 2:
                self.fc_feature_shared_3_target = nn.Linear(feat_shared_dim, feat_shared_dim)
                normal_(self.fc_feature_shared_3_target.weight, 0, std)
                constant_(self.fc_feature_shared_3_target.bias, 0)

            self.fc_feature_target = nn.Linear(feat_shared_dim, feat_frame_dim)
            normal_(self.fc_feature_target.weight, 0, std)
            constant_(self.fc_feature_target.bias, 0)

            self.fc_classifier_target_verb = nn.Linear(feat_frame_dim, num_class[0])
            normal_(self.fc_classifier_target_verb.weight, 0, std)
            constant_(self.fc_classifier_target_verb.bias, 0)
            self.fc_classifier_target_noun = nn.Linear(feat_frame_dim, num_class[1])
            normal_(self.fc_classifier_target_noun.weight, 0, std)
            constant_(self.fc_classifier_target_noun.bias, 0)

        # BN for the above layers
        if self.use_bn is not None:  # S & T: use AdaBN (ICLRW 2017) approach
            self.bn_shared_S = nn.BatchNorm1d(feat_shared_dim)  # BN for the shared layers
            self.bn_shared_T = nn.BatchNorm1d(feat_shared_dim)
            self.bn_source_S = nn.BatchNorm1d(feat_frame_dim)  # BN for the source feature layers
            self.bn_source_T = nn.BatchNorm1d(feat_frame_dim)

        # ------ aggregate frame-based features (frame feature --> video feature) ------#
        if self.frame_aggregation == "rnn":  # 2. rnn
            self.hidden_dim = feat_frame_dim
            if self.rnn_cell == "LSTM":
                self.rnn = nn.LSTM(
                    feat_frame_dim,
                    self.hidden_dim // self.n_directions,
                    self.n_layers,
                    batch_first=True,
                    bidirectional=bool(int(self.n_directions / 2)),
                )
            elif self.rnn_cell == "GRU":
                self.rnn = nn.GRU(
                    feat_frame_dim,
                    self.hidden_dim // self.n_directions,
                    self.n_layers,
                    batch_first=True,
                    bidirectional=bool(int(self.n_directions / 2)),
                )

            # initialization
            for p in range(self.n_layers):
                kaiming_normal_(self.rnn.all_weights[p][0])
                kaiming_normal_(self.rnn.all_weights[p][1])

            self.bn_before_rnn = nn.BatchNorm2d(1)
            self.bn_after_rnn = nn.BatchNorm2d(1)

        elif self.frame_aggregation == "trn":  # 4. TRN (ECCV 2018) ==> fix segment # for both train/val
            self.num_bottleneck = 512
            self.TRN = TRNRelationModule(feat_shared_dim, self.num_bottleneck, self.train_segments)
            self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
            self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)
        elif self.frame_aggregation == "trn-m":  # 4. TRN (ECCV 2018) ==> fix segment # for both train/val
            self.num_bottleneck = 256
            self.TRN = TRNRelationModuleMultiScale(feat_shared_dim, self.num_bottleneck, self.train_segments)
            self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
            self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)

        elif self.frame_aggregation == "temconv":  # 3. temconv

            self.tcl_3_1 = TCL(3, 1)
            self.tcl_5_1 = TCL(5, 1)
            self.bn_1_S = nn.BatchNorm1d(feat_frame_dim)
            self.bn_1_T = nn.BatchNorm1d(feat_frame_dim)

            self.tcl_3_2 = TCL(3, 1)
            self.tcl_5_2 = TCL(5, 2)
            self.bn_2_S = nn.BatchNorm1d(feat_frame_dim)
            self.bn_2_T = nn.BatchNorm1d(feat_frame_dim)

            self.conv_fusion = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=(1, 1), padding=(0, 0)), nn.ReLU(inplace=True),
            )

        # ------ video-level layers (source layers + domain layers) ------#
        if self.frame_aggregation == "avgpool":  # 1. avgpool
            feat_aggregated_dim = feat_shared_dim
        if "trn" in self.frame_aggregation:  # 4. trn
            feat_aggregated_dim = self.num_bottleneck
        elif self.frame_aggregation == "rnn":  # 2. rnn
            feat_aggregated_dim = self.hidden_dim
        elif self.frame_aggregation == "temconv":  # 3. temconv
            feat_aggregated_dim = feat_shared_dim

        feat_video_dim = feat_aggregated_dim

        # 1. source feature layers (video-level)
        # TODO
        self.fc_feature_video_source = nn.Linear(feat_aggregated_dim, feat_video_dim)
        normal_(self.fc_feature_video_source.weight, 0, std)
        constant_(self.fc_feature_video_source.bias, 0)

        self.fc_feature_video_source_2 = nn.Linear(feat_video_dim, feat_video_dim)
        normal_(self.fc_feature_video_source_2.weight, 0, std)
        constant_(self.fc_feature_video_source_2.bias, 0)

        # 2. domain feature layers (video-level)
        self.fc_feature_domain_video = nn.Linear(feat_aggregated_dim, feat_video_dim)
        normal_(self.fc_feature_domain_video.weight, 0, std)
        constant_(self.fc_feature_domain_video.bias, 0)

        # 3. classifiers (video-level)
        self.fc_classifier_video_verb_source = nn.Linear(feat_video_dim, num_class["verb"])
        normal_(self.fc_classifier_video_verb_source.weight, 0, std)
        constant_(self.fc_classifier_video_verb_source.bias, 0)

        self.fc_classifier_video_noun_source = nn.Linear(feat_video_dim, num_class["noun"])
        normal_(self.fc_classifier_video_noun_source.weight, 0, std)
        constant_(self.fc_classifier_video_noun_source.bias, 0)

        if self.ens_DA == "MCD":
            self.fc_classifier_video_source_2 = nn.Linear(
                feat_video_dim, num_class
            )  # second classifier for self-ensembling
            normal_(self.fc_classifier_video_source_2.weight, 0, std)
            constant_(self.fc_classifier_video_source_2.bias, 0)

        self.fc_classifier_domain_video = nn.Linear(feat_video_dim, 2)
        normal_(self.fc_classifier_domain_video.weight, 0, std)
        constant_(self.fc_classifier_domain_video.bias, 0)

        # domain classifier for TRN-M
        if self.frame_aggregation == "trn-m":
            self.relation_domain_classifier_all = nn.ModuleList()
            for i in range(self.train_segments - 1):
                relation_domain_classifier = nn.Sequential(
                    nn.Linear(feat_aggregated_dim, feat_video_dim), nn.ReLU(), nn.Linear(feat_video_dim, 2)
                )
                self.relation_domain_classifier_all += [relation_domain_classifier]

        if self.share_params == "N":
            self.fc_feature_video_target = nn.Linear(feat_aggregated_dim, feat_video_dim)
            normal_(self.fc_feature_video_target.weight, 0, std)
            constant_(self.fc_feature_video_target.bias, 0)
            self.fc_feature_video_target_2 = nn.Linear(feat_video_dim, feat_video_dim)
            normal_(self.fc_feature_video_target_2.weight, 0, std)
            constant_(self.fc_feature_video_target_2.bias, 0)

            self.fc_classifier_video_verb_target = nn.Linear(feat_video_dim, num_class)
            normal_(self.fc_classifier_video_verb_target.weight, 0, std)
            constant_(self.fc_classifier_video_verb_target.bias, 0)

            self.fc_classifier_video_noun_target = nn.Linear(feat_video_dim, num_class)
            normal_(self.fc_classifier_video_noun_target.weight, 0, std)
            constant_(self.fc_classifier_video_noun_target.bias, 0)

        # BN for the above layers
        if self.use_bn is not None:  # S & T: use AdaBN (ICLRW 2017) approach
            self.bn_source_video_S = nn.BatchNorm1d(feat_video_dim)
            self.bn_source_video_T = nn.BatchNorm1d(feat_video_dim)
            self.bn_source_video_2_S = nn.BatchNorm1d(feat_video_dim)
            self.bn_source_video_2_T = nn.BatchNorm1d(feat_video_dim)

        # self.alpha = torch.ones(1)
        if self.use_bn == "AutoDIAL":
            self.alpha = nn.Parameter(self.alpha)

        # ------ attention mechanism ------#
        # conventional attention
        if self.use_attn == "general":
            self.attn_layer = nn.Sequential(
                nn.Linear(feat_aggregated_dim, feat_aggregated_dim), nn.Tanh(), nn.Linear(feat_aggregated_dim, 1)
            )

    # def train(self, mode=True):
    #     # not necessary in our setting
    #     """Override the default train() to freeze the BN parameters"""
    #
    #     super(TA3NTrainerVideo, self).train(mode)
    #     count = 0
    #     if self._enable_pbn:
    #         log_debug("Freezing BatchNorm2D except the first one.")
    #         for m in self.base_model.modules():
    #             if isinstance(m, nn.BatchNorm2d):
    #                 count += 1
    #                 if count >= (2 if self._enable_pbn else 1):
    #                     m.eval()
    #
    #                     # shutdown update in frozen mode
    #                     m.weight.requires_grad = False
    #                     m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_trans_attn(self, pred_domain):
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
        weights = 1 - entropy

        return weights

    def get_general_attn(self, feat):
        num_segments = feat.size()[1]
        feat = feat.view(-1, feat.size()[-1])  # reshape features: 128x4x256 --> (128x4)x256
        weights = self.attn_layer(feat)  # e.g. (128x4)x1
        weights = weights.view(-1, num_segments, weights.size()[-1])  # reshape attention weights: (128x4)x1 --> 128x4x1
        weights = F.softmax(weights, dim=1)  # softmax over segments ==> 128x4x1

        return weights

    def get_attn_feat_frame(self, feat_fc, pred_domain):  # not used for now
        if self.use_attn == "TransAttn":
            weights_attn = self.get_trans_attn(pred_domain)
        elif self.use_attn == "general":
            weights_attn = self.get_general_attn(feat_fc)

        weights_attn = weights_attn.view(-1, 1).repeat(
            1, feat_fc.size()[-1]
        )  # reshape & repeat weights (e.g. 16 x 512)
        feat_fc_attn = (weights_attn + 1) * feat_fc

        return feat_fc_attn

    def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
        if self.use_attn == "TransAttn":
            weights_attn = self.get_trans_attn(pred_domain)
        elif self.use_attn == "general":
            weights_attn = self.get_general_attn(feat_fc)

        weights_attn = weights_attn.view(-1, num_segments - 1, 1).repeat(
            1, 1, feat_fc.size()[-1]
        )  # reshape & repeat weights (e.g. 16 x 4 x 256)
        feat_fc_attn = (weights_attn + 1) * feat_fc

        return feat_fc_attn, weights_attn[:, :, 0]

    def aggregate_frames(self, feat_fc, num_segments, pred_domain):
        feat_fc_video = None
        if self.frame_aggregation == "rnn":
            # 2. RNN
            feat_fc_video = feat_fc.view((-1, num_segments) + feat_fc.size()[-1:])  # reshape for RNN

            # temporal segments and pooling
            len_ts = round(num_segments / self.n_ts)
            num_extra_f = len_ts * self.n_ts - num_segments
            if num_extra_f < 0:  # can remove last frame-level features
                feat_fc_video = feat_fc_video[
                                :, : len_ts * self.n_ts, :
                                ]  # make the temporal length can be divided by n_ts (16 x 25 x 512 --> 16 x 24 x 512)
            elif num_extra_f > 0:  # need to repeat last frame-level features
                feat_fc_video = torch.cat(
                    (feat_fc_video, feat_fc_video[:, -1:, :].repeat(1, num_extra_f, 1)), 1
                )  # make the temporal length can be divided by n_ts (16 x 5 x 512 --> 16 x 6 x 512)

            feat_fc_video = feat_fc_video.view(
                (-1, self.n_ts, len_ts) + feat_fc_video.size()[2:]
            )  # 16 x 6 x 512 --> 16 x 3 x 2 x 512
            feat_fc_video = nn.MaxPool2d(kernel_size=(len_ts, 1))(
                feat_fc_video
            )  # 16 x 3 x 2 x 512 --> 16 x 3 x 1 x 512
            feat_fc_video = feat_fc_video.squeeze(2)  # 16 x 3 x 1 x 512 --> 16 x 3 x 512

            hidden_temp = torch.zeros(
                self.n_layers * self.n_directions, feat_fc_video.size(0), self.hidden_dim // self.n_directions
            ).type_as(feat_fc)

            if self.rnn_cell == "LSTM":
                hidden_init = (hidden_temp, hidden_temp)
            elif self.rnn_cell == "GRU":
                hidden_init = hidden_temp

            self.rnn.flatten_parameters()
            feat_fc_video, hidden_final = self.rnn(feat_fc_video, hidden_init)  # e.g. 16 x 25 x 512

            # get the last feature vector
            feat_fc_video = feat_fc_video[:, -1, :]

        else:
            # 1. averaging
            feat_fc_video = feat_fc.view(
                (-1, 1, num_segments) + feat_fc.size()[-1:]
            )  # reshape based on the segments (e.g. 16 x 1 x 5 x 512)
            if self.use_attn == "TransAttn":  # get the attention weighting
                weights_attn = self.get_trans_attn(pred_domain)
                weights_attn = weights_attn.view(-1, 1, num_segments, 1).repeat(
                    1, 1, 1, feat_fc.size()[-1]
                )  # reshape & repeat weights (e.g. 16 x 1 x 5 x 512)
                feat_fc_video = (weights_attn + 1) * feat_fc_video

            feat_fc_video = nn.AvgPool2d([num_segments, 1])(feat_fc_video)  # e.g. 16 x 1 x 1 x 512
            feat_fc_video = feat_fc_video.squeeze(1).squeeze(1)  # e.g. 16 x 512

        return feat_fc_video

    def final_output(self, pred, pred_video, num_segments):
        if self.baseline_type == "video":
            base_out = pred_video
        else:
            base_out = pred

        if not self.before_softmax:
            base_out = (self.softmax(base_out[0]), self.softmax(base_out[1]))
        output = base_out

        if self.baseline_type == "tsn":
            if self.reshape:
                base_out = (
                    base_out[0].view((-1, num_segments) + base_out[0].size()[1:]),
                    base_out[1].view((-1, num_segments) + base_out[1].size()[1:]),
                )  # e.g. 16 x 3 x 12 (3 segments)
            output = (base_out[0].mean(1), base_out[1].mean(1))  # e.g. 16 x 12

        return output

    def domain_classifier_frame(self, feat, beta):
        feat_fc_domain_frame = GradReverse.apply(feat, beta[2])
        feat_fc_domain_frame = self.fc_feature_domain(feat_fc_domain_frame)
        feat_fc_domain_frame = self.relu(feat_fc_domain_frame)
        pred_fc_domain_frame = self.fc_classifier_domain(feat_fc_domain_frame)

        return pred_fc_domain_frame

    def domain_classifier_video(self, feat_video, beta):
        feat_fc_domain_video = GradReverse.apply(feat_video, beta[1])
        feat_fc_domain_video = self.fc_feature_domain_video(feat_fc_domain_video)
        feat_fc_domain_video = self.relu(feat_fc_domain_video)
        pred_fc_domain_video = self.fc_classifier_domain_video(feat_fc_domain_video)

        return pred_fc_domain_video

    def domain_classifier_relation(self, feat_relation, beta):
        # 128x4x256 --> (128x4)x2
        pred_fc_domain_relation_video = None
        for i in range(len(self.relation_domain_classifier_all)):
            feat_relation_single = feat_relation[:, i, :].squeeze(1)  # 128x1x256 --> 128x256
            feat_fc_domain_relation_single = GradReverse.apply(
                feat_relation_single, beta[0]
            )  # the same beta for all relations (for now)

            pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)

            if pred_fc_domain_relation_video is None:
                pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1, 1, 2)
            else:
                pred_fc_domain_relation_video = torch.cat(
                    (pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1, 1, 2)), 1
                )

        pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1, 2)

        return pred_fc_domain_relation_video

    def domainAlign(self, input_S, input_T, is_train, name_layer, alpha, num_segments, dim):
        input_S = input_S.view(
            (-1, dim, num_segments) + input_S.size()[-1:]
        )  # reshape based on the segments (e.g. 80 x 512 --> 16 x 1 x 5 x 512)
        input_T = input_T.view((-1, dim, num_segments) + input_T.size()[-1:])  # reshape based on the segments

        # clamp alpha
        alpha = max(alpha, 0.5)

        # rearange source and target data
        num_S_1 = int(round(input_S.size(0) * alpha))
        num_S_2 = input_S.size(0) - num_S_1
        num_T_1 = int(round(input_T.size(0) * alpha))
        num_T_2 = input_T.size(0) - num_T_1

        if is_train and num_S_2 > 0 and num_T_2 > 0:
            input_source = torch.cat((input_S[:num_S_1], input_T[-num_T_2:]), 0)
            input_target = torch.cat((input_T[:num_T_1], input_S[-num_S_2:]), 0)
        else:
            input_source = input_S
            input_target = input_T

        # adaptive BN
        input_source = input_source.view(
            (-1,) + input_source.size()[-1:]
        )  # reshape to feed BN (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
        input_target = input_target.view((-1,) + input_target.size()[-1:])

        if name_layer == "shared":
            input_source_bn = self.bn_shared_S(input_source)
            input_target_bn = self.bn_shared_T(input_target)
        elif "trn" in name_layer:
            input_source_bn = self.bn_trn_S(input_source)
            input_target_bn = self.bn_trn_T(input_target)
        elif name_layer == "temconv_1":
            input_source_bn = self.bn_1_S(input_source)
            input_target_bn = self.bn_1_T(input_target)
        elif name_layer == "temconv_2":
            input_source_bn = self.bn_2_S(input_source)
            input_target_bn = self.bn_2_T(input_target)

        input_source_bn = input_source_bn.view(
            (-1, dim, num_segments) + input_source_bn.size()[-1:]
        )  # reshape back (e.g. 80 x 512 --> 16 x 1 x 5 x 512)
        input_target_bn = input_target_bn.view((-1, dim, num_segments) + input_target_bn.size()[-1:])  #

        # rearange back to the original order of source and target data (since target may be unlabeled)
        if is_train and num_S_2 > 0 and num_T_2 > 0:
            input_source_bn = torch.cat((input_source_bn[:num_S_1], input_target_bn[-num_S_2:]), 0)
            input_target_bn = torch.cat((input_target_bn[:num_T_1], input_source_bn[-num_T_2:]), 0)

        # reshape for frame-level features
        if name_layer == "shared" or name_layer == "trn_sum":
            input_source_bn = input_source_bn.view(
                (-1,) + input_source_bn.size()[-1:]
            )  # (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
            input_target_bn = input_target_bn.view((-1,) + input_target_bn.size()[-1:])
        elif name_layer == "trn":
            input_source_bn = input_source_bn.view(
                (-1, num_segments) + input_source_bn.size()[-1:]
            )  # (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
            input_target_bn = input_target_bn.view((-1, num_segments) + input_target_bn.size()[-1:])

        return input_source_bn, input_target_bn

    def forward(self, input_source, input_target, beta, mu, is_train=True, reverse=True):
        batch_source = input_source.size()[0]
        batch_target = input_target.size()[0]
        num_segments = self.train_segments if is_train else self.val_segments
        # sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        sample_len = self.new_length
        feat_all_source = []
        feat_all_target = []
        pred_domain_all_source = []
        pred_domain_all_target = []
        # log_output("input_source: ", input_source)
        # log_output("input_target: ", input_target)
        # input_data is a list of tensors --> need to do pre-processing
        feat_base_source = input_source.view(-1, input_source.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048
        feat_base_target = input_target.view(-1, input_target.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048

        # === shared layers ===#
        # need to separate BN for source & target ==> otherwise easy to overfit to source data
        if self.add_fc < 1:
            raise ValueError("not enough fc layer")

        feat_fc_source = self.fc_feature_shared_source(feat_base_source)
        feat_fc_target = (
            self.fc_feature_shared_target(feat_base_target)
            if self.share_params == "N"
            else self.fc_feature_shared_source(feat_base_target)
        )
        # log_output("feat_fc_source: ", feat_fc_source)
        # log_output("feat_fc_target: ", feat_fc_target)
        # adaptive BN
        # if self.use_bn is not None:
        #     feat_fc_source, feat_fc_target = self.domainAlign(feat_fc_source, feat_fc_target, is_train, 'shared',
        #                                                       self.alpha.item(), num_segments, 1)

        feat_fc_source = self.relu(feat_fc_source)
        feat_fc_target = self.relu(feat_fc_target)
        feat_fc_source = self.dropout_i(feat_fc_source)
        feat_fc_target = self.dropout_i(feat_fc_target)
        # log_output("feat_fc_source: ", feat_fc_source)
        # log_output("feat_fc_target: ", feat_fc_target)

        # feat_fc = self.dropout_i(feat_fc)
        feat_all_source.append(
            feat_fc_source.view((batch_source, num_segments) + feat_fc_source.size()[-1:])
        )  # reshape ==> 1st dim is the batch size
        feat_all_target.append(feat_fc_target.view((batch_target, num_segments) + feat_fc_target.size()[-1:]))

        # if self.add_fc > 1:
        #     feat_fc_source = self.fc_feature_shared_2_source(feat_fc_source)
        #     feat_fc_target = self.fc_feature_shared_2_target(
        #         feat_fc_target) if self.share_params == 'N' else self.fc_feature_shared_2_source(feat_fc_target)
        #
        #     feat_fc_source = self.relu(feat_fc_source)
        #     feat_fc_target = self.relu(feat_fc_target)
        #     feat_fc_source = self.dropout_i(feat_fc_source)
        #     feat_fc_target = self.dropout_i(feat_fc_target)
        #
        #     feat_all_source.append(feat_fc_source.view(
        #         (batch_source, num_segments) + feat_fc_source.size()[-1:]))  # reshape ==> 1st dim is the batch size
        #     feat_all_target.append(feat_fc_target.view((batch_target, num_segments) + feat_fc_target.size()[-1:]))
        #
        # if self.add_fc > 2:
        #     feat_fc_source = self.fc_feature_shared_3_source(feat_fc_source)
        #     feat_fc_target = self.fc_feature_shared_3_target(
        #         feat_fc_target) if self.share_params == 'N' else self.fc_feature_shared_3_source(feat_fc_target)
        #
        #     feat_fc_source = self.relu(feat_fc_source)
        #     feat_fc_target = self.relu(feat_fc_target)
        #     feat_fc_source = self.dropout_i(feat_fc_source)
        #     feat_fc_target = self.dropout_i(feat_fc_target)
        #
        #     feat_all_source.append(feat_fc_source.view(
        #         (batch_source, num_segments) + feat_fc_source.size()[-1:]))  # reshape ==> 1st dim is the batch size
        #     feat_all_target.append(feat_fc_target.view((batch_target, num_segments) + feat_fc_target.size()[-1:]))

        # === adversarial branch (frame-level) ===#
        pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, beta)
        pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, beta)
        # log_output("pred_fc_domain_frame_source: ", pred_fc_domain_frame_source)
        # log_output("pred_fc_domain_frame_target: ", pred_fc_domain_frame_target)

        pred_domain_all_source.append(
            pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:])
        )
        pred_domain_all_target.append(
            pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:])
        )

        # if self.use_attn_frame is not None:  # attend the frame-level features only
        #     feat_fc_source = self.get_attn_feat_frame(feat_fc_source, pred_fc_domain_frame_source)
        #     feat_fc_target = self.get_attn_feat_frame(feat_fc_target, pred_fc_domain_frame_target)
        #     log_output("feat_fc_source (after attention): ", feat_fc_source)
        #     log_output("feat_fc_target (after attention): ", feat_fc_target)

        # === source layers (frame-level) ===#

        pred_fc_source = (
            self.fc_classifier_source_verb(feat_fc_source),
            self.fc_classifier_source_noun(feat_fc_source),
        )
        pred_fc_target = (
            self.fc_classifier_target_verb(feat_fc_target)
            if self.share_params == "N"
            else self.fc_classifier_source_verb(feat_fc_target),
            self.fc_classifier_target_noun(feat_fc_target)
            if self.share_params == "N"
            else self.fc_classifier_source_noun(feat_fc_target),
        )

        # log_output("pred_fc_source: ", pred_fc_source)
        # log_output("pred_fc_target: ", pred_fc_target)
        # if self.baseline_type == 'frame':
        #     feat_all_source.append(pred_fc_source[0].view(
        #         (batch_source, num_segments) + pred_fc_source[0].size()[-1:]))  # reshape ==> 1st dim is the batch size
        #     feat_all_target.append(pred_fc_target[0].view((batch_target, num_segments) + pred_fc_target[0].size()[-1:]))

        ### aggregate the frame-based features to video-based features ###
        if self.frame_aggregation == "avgpool" or self.frame_aggregation == "rnn":
            feat_fc_video_source = self.aggregate_frames(feat_fc_source, num_segments, pred_fc_domain_frame_source)
            feat_fc_video_target = self.aggregate_frames(feat_fc_target, num_segments, pred_fc_domain_frame_target)

            attn_relation_source = feat_fc_video_source[
                                   :, 0
                                   ]  # assign random tensors to attention values to avoid runtime error
            attn_relation_target = feat_fc_video_target[
                                   :, 0
                                   ]  # assign random tensors to attention values to avoid runtime error

        elif "trn" in self.frame_aggregation:
            feat_fc_video_source = feat_fc_source.view(
                (-1, num_segments) + feat_fc_source.size()[-1:]
            )  # reshape based on the segments (e.g. 640x512 --> 128x5x512)
            feat_fc_video_target = feat_fc_target.view(
                (-1, num_segments) + feat_fc_target.size()[-1:]
            )  # reshape based on the segments (e.g. 640x512 --> 128x5x512)

            feat_fc_video_relation_source = self.TRN(
                feat_fc_video_source
            )  # 128x5x512 --> 128x5x256 (256-dim. relation feature vectors x 5)
            feat_fc_video_relation_target = self.TRN(feat_fc_video_target)
            # log_output("feat_fc_video_relation_source: ", feat_fc_video_relation_source)
            # log_output("feat_fc_video_relation_target: ", feat_fc_video_relation_target)
            # adversarial branch
            pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source, beta)
            pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target, beta)
            # log_output("pred_fc_domain_video_relation_source: ", pred_fc_domain_video_relation_source)
            # log_output("pred_fc_domain_video_relation_target: ", pred_fc_domain_video_relation_target)

            # transferable attention
            if self.use_attn is not None:  # get the attention weighting
                feat_fc_video_relation_source, attn_relation_source = self.get_attn_feat_relation(
                    feat_fc_video_relation_source, pred_fc_domain_video_relation_source, num_segments
                )
                feat_fc_video_relation_target, attn_relation_target = self.get_attn_feat_relation(
                    feat_fc_video_relation_target, pred_fc_domain_video_relation_target, num_segments
                )
            else:
                attn_relation_source = feat_fc_video_relation_source[
                                       :, :, 0
                                       ]  # assign random tensors to attention values to avoid runtime error
                attn_relation_target = feat_fc_video_relation_target[
                                       :, :, 0
                                       ]  # assign random tensors to attention values to avoid runtime error

            # sum up relation features (ignore 1-relation)
            feat_fc_video_source = torch.sum(feat_fc_video_relation_source, 1)
            feat_fc_video_target = torch.sum(feat_fc_video_relation_target, 1)
            # log_output("feat_fc_video_source: ", feat_fc_video_source)
            # log_output("feat_fc_video_target: ", feat_fc_video_target)

        elif self.frame_aggregation == "temconv":  # DA operation inside temconv
            feat_fc_video_source = feat_fc_source.view(
                (-1, 1, num_segments) + feat_fc_source.size()[-1:]
            )  # reshape based on the segments
            feat_fc_video_target = feat_fc_target.view(
                (-1, 1, num_segments) + feat_fc_target.size()[-1:]
            )  # reshape based on the segments

            # 1st TCL
            feat_fc_video_source_3_1 = self.tcl_3_1(feat_fc_video_source)
            feat_fc_video_target_3_1 = self.tcl_3_1(feat_fc_video_target)

            if self.use_bn is not None:
                feat_fc_video_source_3_1, feat_fc_video_target_3_1 = self.domainAlign(
                    feat_fc_video_source_3_1,
                    feat_fc_video_target_3_1,
                    is_train,
                    "temconv_1",
                    self.alpha.item(),
                    num_segments,
                    1,
                )

            feat_fc_video_source = self.relu(feat_fc_video_source_3_1)  # 16 x 1 x 5 x 512
            feat_fc_video_target = self.relu(feat_fc_video_target_3_1)  # 16 x 1 x 5 x 512

            feat_fc_video_source = nn.AvgPool2d(kernel_size=(num_segments, 1))(feat_fc_video_source)  # 16 x 4 x 1 x 512
            feat_fc_video_target = nn.AvgPool2d(kernel_size=(num_segments, 1))(feat_fc_video_target)  # 16 x 4 x 1 x 512

            feat_fc_video_source = feat_fc_video_source.squeeze(1).squeeze(1)  # e.g. 16 x 512
            feat_fc_video_target = feat_fc_video_target.squeeze(1).squeeze(1)  # e.g. 16 x 512

        if self.baseline_type == "video":
            feat_all_source.append(feat_fc_video_source.view((batch_source,) + feat_fc_video_source.size()[-1:]))
            feat_all_target.append(feat_fc_video_target.view((batch_target,) + feat_fc_video_target.size()[-1:]))

        # === source layers (video-level) ===#
        feat_fc_video_source = self.dropout_v(feat_fc_video_source)
        feat_fc_video_target = self.dropout_v(feat_fc_video_target)
        # log_output("feat_fc_video_source: ", feat_fc_video_source)
        # log_output("feat_fc_video_target: ", feat_fc_video_target)

        # if reverse:
        #     feat_fc_video_source = GradReverse.apply(feat_fc_video_source, mu)
        #     feat_fc_video_target = GradReverse.apply(feat_fc_video_target, mu)

        # log_output("feat_fc_video_source: ", feat_fc_video_source)
        # log_output("feat_fc_video_target: ", feat_fc_video_target)

        pred_fc_video_source = (
            self.fc_classifier_video_verb_source(feat_fc_video_source),
            self.fc_classifier_video_noun_source(feat_fc_video_source),
        )
        pred_fc_video_target = (
            self.fc_classifier_video_verb_target(feat_fc_video_target)
            if self.share_params == "N"
            else self.fc_classifier_video_verb_source(feat_fc_video_target),
            self.fc_classifier_video_noun_target(feat_fc_video_target)
            if self.share_params == "N"
            else self.fc_classifier_video_noun_source(feat_fc_video_target),
        )

        # log_output("pred_fc_video_source: ", pred_fc_video_source)
        # log_output("pred_fc_video_target: ", pred_fc_video_target)

        if self.baseline_type == "video":  # only store the prediction from classifier 1 (for now)
            feat_all_source.append(pred_fc_video_source[0].view((batch_source,) + pred_fc_video_source[0].size()[-1:]))
            feat_all_target.append(pred_fc_video_target[0].view((batch_target,) + pred_fc_video_target[0].size()[-1:]))
            feat_all_source.append(pred_fc_video_source[1].view((batch_source,) + pred_fc_video_source[1].size()[-1:]))
            feat_all_target.append(pred_fc_video_target[1].view((batch_target,) + pred_fc_video_target[1].size()[-1:]))

        # === adversarial branch (video-level) ===#
        pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, beta)
        pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, beta)
        # log_output("pred_fc_domain_video_source: ", pred_fc_domain_video_source)
        # log_output("pred_fc_domain_video_target: ", pred_fc_domain_video_target)

        pred_domain_all_source.append(
            pred_fc_domain_video_source.view((batch_source,) + pred_fc_domain_video_source.size()[-1:])
        )
        pred_domain_all_target.append(
            pred_fc_domain_video_target.view((batch_target,) + pred_fc_domain_video_target.size()[-1:])
        )

        # video relation-based discriminator
        if self.frame_aggregation == "trn-m":
            num_relation = feat_fc_video_relation_source.size()[1]
            pred_domain_all_source.append(
                pred_fc_domain_video_relation_source.view(
                    (batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:]
                )
            )
            pred_domain_all_target.append(
                pred_fc_domain_video_relation_target.view(
                    (batch_target, num_relation) + pred_fc_domain_video_relation_target.size()[-1:]
                )
            )
        else:
            pred_domain_all_source.append(
                pred_fc_domain_video_source
            )  # if not trn-m, add dummy tensors for relation features
            pred_domain_all_target.append(pred_fc_domain_video_target)

        # === final output ===#
        output_source = self.final_output(
            pred_fc_source, pred_fc_video_source, num_segments
        )  # select output from frame or video prediction
        output_target = self.final_output(pred_fc_target, pred_fc_video_target, num_segments)

        output_source_2 = output_source
        output_target_2 = output_target

        # if self.ens_DA == 'MCD':
        #     pred_fc_video_source_2 = self.fc_classifier_video_source_2(feat_fc_video_source)
        #     pred_fc_video_target_2 = self.fc_classifier_video_target_2(
        #         feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_source_2(
        #         feat_fc_video_target)
        #     output_source_2 = self.final_output(pred_fc_source, pred_fc_video_source_2, num_segments)
        #     output_target_2 = self.final_output(pred_fc_target, pred_fc_video_target_2, num_segments)

        return (
            attn_relation_source,
            output_source,
            output_source_2,
            pred_domain_all_source[::-1],
            feat_all_source[::-1],
            attn_relation_target,
            output_target,
            output_target_2,
            pred_domain_all_target[::-1],
            feat_all_target[::-1],
        )
        # reverse the order of feature list due to some multi-gpu issues

    def configure_optimizers(self):
        """Automatically called by lightning to get the optimizer for training"""

        if self._optimizer_params["type"] == "SGD":
            # log_info("using SGD")
            # optimizer = torch.optim.SGD(
            #     self.parameters(), self._init_lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True
            # )
            optimizer = torch.optim.SGD(self.parameters(), lr=self._init_lr, **self._optimizer_params["optim_params"], )
        elif self._optimizer_params["type"] == "Adam":
            # log_info("using Adam")
            # optimizer = torch.optim.Adam(self.parameters(), self._init_lr, weight_decay=self.weight_decay)
            optimizer = torch.optim.Adam(self.parameters(), lr=self._init_lr,
                                         **self._optimizer_params["optim_params"], )
        else:
            pass
            # log_error("optimizer not support or specified!!!")

        return optimizer

    def adjust_learning_rate(self, optimizer, decay):
        """Sets the learning rate to the initial LR decayed by 10 """
        for param_group in optimizer.param_groups:
            param_group["lr"] /= decay

    def adjust_learning_rate_loss(self, optimizer, decay, stat_current, stat_previous, op):
        ops = {
            ">": (lambda x, y: x > y),
            "<": (lambda x, y: x < y),
            ">=": (lambda x, y: x >= y),
            "<=": (lambda x, y: x <= y),
        }
        if ops[op](stat_current, stat_previous):
            for param_group in optimizer.param_groups:
                param_group["lr"] /= decay

    def adjust_learning_rate_dann(self, optimizer, p):
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr / (1.0 + 10 * p) ** 0.75

    def get_parameters_watch_list(self):
        """Update this list for parameters to watch while training (ie log with MLFlow)"""

        return {
            "alpha": self.alpha,
            "beta": self.beta[0],
            "mu": self.mu,
            "last_epoch": self.current_epoch,
        }

    def compute_loss(self, batch, split_name="V"):
        ((source_data, source_label, _), (target_data, target_label, _)) = batch

        source_size_ori = source_data.size()  # original shape
        target_size_ori = target_data.size()  # original shape
        # source_size_ori = source_data.shape  # original shape
        # target_size_ori = target_data.shape  # original shape
        batch_source_ori = source_size_ori[0]
        batch_target_ori = target_size_ori[0]

        # add dummy tensors to keep the same batch size for each epoch (for the last epoch)

        # print("batch_source_ori: {}".format(batch_source_ori))

        if batch_source_ori < self._batch_size:
            source_data_dummy = torch.zeros(
                self._batch_size - batch_source_ori, source_size_ori[1], source_size_ori[2]
            ).type_as(source_data)
            source_data = torch.cat((source_data, source_data_dummy))
        if batch_target_ori < self._target_batch_size:
            target_data_dummy = torch.zeros(
                self._target_batch_size - batch_target_ori, target_size_ori[1], target_size_ori[2]
            ).type_as(target_data)
            target_data = torch.cat((target_data, target_data_dummy))

        source_label_verb = source_label[0]  # pytorch 0.4.X
        source_label_noun = source_label[1]  # pytorch 0.4.X

        target_label_verb = target_label[0]  # pytorch 0.4.X
        target_label_noun = target_label[1]  # pytorch 0.4.X

        if self.baseline_type == "frame":
            source_label_verb_frame = (
                source_label_verb.unsqueeze(1).repeat(1, self.train_segments).view(-1)
            )  # expand the size for all the frames
            source_label_noun_frame = (
                source_label_noun.unsqueeze(1).repeat(1, self.train_segments).view(-1)
            )  # expand the size for all the frames
            target_label_verb_frame = target_label_verb.unsqueeze(1).repeat(1, self.train_segments).view(-1)
            target_label_noun_frame = target_label_noun.unsqueeze(1).repeat(1, self.train_segments).view(-1)

        label_source_verb = (
            source_label_verb_frame if self.baseline_type == "frame" else source_label_verb
        )  # determine the label for calculating the loss function
        label_target_verb = target_label_verb_frame if self.baseline_type == "frame" else target_label_verb

        label_source_noun = (
            source_label_noun_frame if self.baseline_type == "frame" else source_label_noun
        )  # determine the label for calculating the loss function
        label_target_noun = target_label_noun_frame if self.baseline_type == "frame" else target_label_noun

        if split_name != "T":
            self.beta_new = self.beta

        (
            attn_source,
            out_source,
            out_source_2,
            pred_domain_source,
            feat_source,
            attn_target,
            out_target,
            out_target_2,
            pred_domain_target,
            feat_target,
        ) = self.forward(source_data, target_data, self.beta_new, self.mu, is_train=True, reverse=False)

        attn_source, out_source, out_source_2, pred_domain_source, feat_source = self.removeDummy(
            attn_source, out_source, out_source_2, pred_domain_source, feat_source, batch_source_ori
        )
        attn_target, out_target, out_target_2, pred_domain_target, feat_target = self.removeDummy(
            attn_target, out_target, out_target_2, pred_domain_target, feat_target, batch_target_ori
        )
        # ====== calculate the loss function ======#
        # 1. calculate the classification loss

        # print(split_name)

        # out_verb = out_source[0]
        # out_noun = out_source[1]
        # target_out_verb = out_target[0]
        # target_out_noun = out_target[1]
        # label_verb = label_source_verb
        # label_noun = label_source_noun

        # loss_verb = self.criterion(out_verb, label_verb)
        # loss_noun = self.criterion(out_noun, label_noun)
        src_loss_verb = self.criterion(out_source[0], label_source_verb)
        src_loss_noun = self.criterion(out_source[1], label_source_noun)
        if self.train_metric == "all":
            src_task_loss = 0.5 * (src_loss_verb + src_loss_noun)
        elif self.train_metric == "noun":
            src_task_loss = src_loss_noun  # 0.5*(loss_verb+loss_noun)
        elif self.train_metric == "verb":
            src_task_loss = src_loss_verb  # 0.5*(loss_verb+loss_noun)
        else:
            raise Exception("invalid metric to train")

        # 2. calculate the loss for DA
        # (I) discrepancy-based approach: discrepancy loss
        # if self.dis_DA is not None and self.use_target is not None:
        #     loss_discrepancy = 0
        #
        #     kernel_muls = [2.0] * 2
        #     kernel_nums = [2, 5]
        #     fix_sigma_list = [None] * 2
        #
        #     if self.dis_DA == "JAN":
        #         # ignore the features from shared layers
        #         feat_source_sel = feat_source[: -self.add_fc]
        #         feat_target_sel = feat_target[: -self.add_fc]
        #
        #         size_loss = min(feat_source_sel[0].size(0), feat_target_sel[0].size(0))  # choose the smaller number
        #         feat_source_sel = [feat[:size_loss] for feat in feat_source_sel]
        #         feat_target_sel = [feat[:size_loss] for feat in feat_target_sel]
        #
        #         loss_discrepancy += JAN(
        #             feat_source_sel,
        #             feat_target_sel,
        #             kernel_muls=kernel_muls,
        #             kernel_nums=kernel_nums,
        #             fix_sigma_list=fix_sigma_list,
        #             ver=2,
        #         )
        #
        #     else:
        #         # extend the parameter list for shared layers
        #         kernel_muls.extend([kernel_muls[-1]] * self.add_fc)
        #         kernel_nums.extend([kernel_nums[-1]] * self.add_fc)
        #         fix_sigma_list.extend([fix_sigma_list[-1]] * self.add_fc)
        #
        #         for l in range(
        #             0, self.add_fc + 2
        #         ):  # loss from all the features (+2 because of frame-aggregation layer + final fc layer)
        #             if self.place_dis[l] == "Y":
        #                 # select the data for calculating the loss (make sure source # == target #)
        #                 size_loss = min(feat_source[l].size(0), feat_target[l].size(0))  # choose the smaller number
        #                 # select
        #                 feat_source_sel = feat_source[l][:size_loss]
        #                 feat_target_sel = feat_target[l][:size_loss]
        #
        #                 # break into multiple batches to avoid "out of memory" issue
        #                 size_batch = min(256, feat_source_sel.size(0))
        #                 feat_source_sel = feat_source_sel.view((-1, size_batch) + feat_source_sel.size()[1:])
        #                 feat_target_sel = feat_target_sel.view((-1, size_batch) + feat_target_sel.size()[1:])
        #
        #             # if self.dis_DA == 'CORAL':
        #             # 	losses_coral = [CORAL(feat_source_sel[t], feat_target_sel[t]) for t in range(feat_source_sel.size(0))]
        #             # 	loss_coral = sum(losses_coral)/len(losses_coral)
        #             # 	loss_discrepancy += loss_coral
        #             # elif self.dis_DA == 'DAN':
        #             # 	losses_mmd = [mmd_rbf(feat_source_sel[t], feat_target_sel[t], kernel_mul=kernel_muls[l], kernel_num=kernel_nums[l], fix_sigma=fix_sigma_list[l], ver=2) for t in range(feat_source_sel.size(0))]
        #             # 	loss_mmd = sum(losses_mmd) / len(losses_mmd)
        #
        #             # 	loss_discrepancy += loss_mmd
        #             # else:
        #             # 	raise NameError('not in dis_DA!!!')
        #     loss += self.alpha * loss_discrepancy

        # (II) adversarial discriminative model: adversarial loss
        loss_adversarial = 0
        if split_name == "T":
            if self.adv_DA is not None and self.use_target is not None:
                loss_adversarial = 0
                pred_domain_all = []
                pred_domain_target_all = []

                for l in range(len(self.place_adv)):
                    if self.place_adv[l] == "Y":

                        # reshape the features (e.g. 128x5x2 --> 640x2)
                        pred_domain_source_single = pred_domain_source[l].view(-1, pred_domain_source[l].size()[-1])
                        pred_domain_target_single = pred_domain_target[l].view(-1, pred_domain_target[l].size()[-1])

                        # prepare domain labels
                        source_domain_label = torch.zeros(pred_domain_source_single.size(0)).type_as(source_data).long()
                        target_domain_label = torch.ones(pred_domain_target_single.size(0)).type_as(source_data).long()
                        domain_label = torch.cat((source_domain_label, target_domain_label), 0)

                        pred_domain = torch.cat((pred_domain_source_single, pred_domain_target_single), 0)
                        pred_domain_all.append(pred_domain)
                        pred_domain_target_all.append(pred_domain_target_single)

                        if self.pred_normalize == "Y":  # use the uncertainly method (in construction......)
                            pred_domain = pred_domain / pred_domain.var().log()
                        loss_adversarial_single = self.criterion_domain(pred_domain, domain_label)

                        loss_adversarial += loss_adversarial_single

                src_task_loss += loss_adversarial

        # (III) other loss
        # 1. entropy loss for target data
        # if self.add_loss_DA == "target_entropy" and self.use_target is not None:
        #     loss_entropy_verb = losses.cross_entropy_soft(out_target[0])
        #     loss_entropy_noun = losses.cross_entropy_soft(out_target[1])
        #
        #     if self.train_metric == "all":
        #         loss += self.gamma * 0.5 * (loss_entropy_verb + loss_entropy_noun)
        #     elif self.train_metric == "noun":
        #         loss += self.gamma * loss_entropy_noun
        #     elif self.train_metric == "verb":
        #         loss += self.gamma * loss_entropy_verb
        #     else:
        #         raise Exception("invalid metric to train")
        # # loss += gamma * 0.5*(loss_entropy_verb+loss_entropy_noun)

            # 3. attentive entropy loss
            if self.add_loss_DA == "attentive_entropy" and self.use_attn is not None and self.use_target is not None:
                loss_entropy_verb = losses.attentive_entropy(torch.cat((out_source[0], out_target[0]), 0), pred_domain_all[1])
                loss_entropy_noun = losses.attentive_entropy(torch.cat((out_source[1], out_target[1]), 0), pred_domain_all[1])

                if self.train_metric == "all":
                    src_task_loss += self.gamma * 0.5 * (loss_entropy_verb + loss_entropy_noun)
                elif self.train_metric == "noun":
                    src_task_loss += self.gamma * loss_entropy_noun
                elif self.train_metric == "verb":
                    src_task_loss += self.gamma * loss_entropy_verb
                else:
                    raise Exception("invalid metric to train")
        # loss += gamma * 0.5*(loss_entropy_verb + loss_entropy_noun)
        # measure accuracy and record loss
        # pred_verb = out_verb
        # prec1_verb, prec5_verb = self.accuracy(pred_verb.data, label_verb, topk=(1, 5))
        # pred_noun = out_noun
        # prec1_noun, prec5_noun = self.accuracy(pred_noun.data, label_noun, topk=(1, 5))
        # prec1_action, prec5_action = self.multitask_accuracy(
        #     (pred_verb.data, pred_noun.data), (label_verb, label_noun), topk=(1, 5)
        # )
        # prec1_action, prec5_action = torch.tensor(prec1_action), torch.tensor(prec5_action)

        prec1_src_verb, prec5_src_verb = self.accuracy(out_source[0].data, label_source_verb, topk=(1, 5))
        prec1_src_noun, prec5_src_noun = self.accuracy(out_source[1].data, label_source_noun, topk=(1, 5))
        prec1_src_action, prec5_src_action = self.multitask_accuracy(
            (out_source[0].data, out_source[1].data), (label_source_verb, label_source_noun), topk=(1, 5)
        )
        prec1_src_action, prec5_src_action = torch.tensor(prec1_src_action), torch.tensor(prec5_src_action)

        prec1_tgt_verb, prec5_tgt_verb = self.accuracy(out_target[0].data, label_target_verb, topk=(1, 5))
        prec1_tgt_noun, prec5_tgt_noun = self.accuracy(out_target[1].data, label_target_noun, topk=(1, 5))
        prec1_tgt_action, prec5_tgt_action = self.multitask_accuracy(
            (out_target[0].data, out_target[1].data), (label_target_verb, label_target_noun), topk=(1, 5)
        )
        prec1_tgt_action, prec5_tgt_action = torch.tensor(prec1_tgt_action), torch.tensor(prec5_tgt_action)

        # measure elapsed time
        # batch_time = time.time() - self.end
        # self.end = time.time()

        # # Pred normalise
        # if self.pred_normalize == "Y":  # use the uncertainly method (in contruction...)
        #     out_source = out_source / out_source.var().log()
        #     out_target = out_target / out_target.var().log()

        # ======= return log_metrics ======#

        log_metrics = {
            f"{split_name}_verb_source_acc": prec1_src_verb,
            f"{split_name}_noun_source_acc": prec1_src_noun,
            f"{split_name}_verb_target_acc": prec1_tgt_verb,
            f"{split_name}_noun_target_acc": prec1_tgt_noun,
            f"{split_name}_verb_source_top1_acc": prec1_src_verb,
            f"{split_name}_verb_source_top5_acc": prec5_src_verb,
            f"{split_name}_noun_source_top1_acc": prec1_src_noun,
            f"{split_name}_noun_source_top5_acc": prec5_src_noun,
            f"{split_name}_action_source_top1_acc": prec1_src_action,
            f"{split_name}_action_source_top5_acc": prec5_src_action,
            f"{split_name}_verb_target_top1_acc": prec1_tgt_verb,
            f"{split_name}_verb_target_top5_acc": prec5_tgt_verb,
            f"{split_name}_noun_target_top1_acc": prec1_tgt_noun,
            f"{split_name}_noun_target_top5_acc": prec5_tgt_noun,
            f"{split_name}_action_target_top1_acc": prec1_tgt_action,
            f"{split_name}_action_target_top5_acc": prec5_tgt_action,
            f"{split_name}_domain_acc": prec1_src_verb,
        }

        # log_metrics = {
        #     "Loss Total": loss,
        #     "Prec@1 Verb": prec1_verb,
        #     "Prec@5 Verb": prec5_verb,
        #     "Prec@1 Noun": prec1_noun,
        #     "Prec@5 Noun": prec5_noun,
        #     "Prec@1 Action": prec1_action,
        #     "Prec@5 Action": prec5_action,
        # }

        # log_output("loss total: ", loss)
        # log_output("Prec@1 Verb: ", prec1_verb)
        # log_output("Prec@5 Verb: ", prec5_verb)
        # log_output("Prec@1 Noun: ", prec1_noun)
        # log_output("Prec@5 Noun: ", prec5_noun)
        # log_output("Prec@1 Action: ", prec1_action)
        # log_output("Prec@5 Action: ", prec5_action)

        # return loss_classification, loss_adversarial, log_metrics
        return src_task_loss, src_task_loss, loss_adversarial, log_metrics

    def _update_batch_epoch_factors(self, batch_id):
        # setup hyperparameters
        loss_c_current = 999  # random large number
        loss_c_previous = 999  # random large number

        start_steps = len(self.trainer.train_dataloader)
        total_steps = self.nb_adapt_epochs * len(self.trainer.train_dataloader)
        p = float(self.global_step + start_steps) / total_steps
        self.beta_dann = 2.0 / (1.0 + np.exp(-1.0 * p)) - 1
        self.beta = [
            self.beta_dann if self.beta[i] < 0 else self.beta[i] for i in range(len(self.beta))
        ]  # replace the default beta if value < 0
        if self.dann_warmup:
            self.beta_new = [self.beta_dann * self.beta[i] for i in range(len(self.beta))]
        else:
            self.beta_new = self.beta

        # print("nb_adapt_epochs: {}, len: {}".format(self.nb_adapt_epochs, len(self.trainer.train_dataloader)))
        # print("i+start_steps: {}, total_steps: {}, p :{}, beta_new: {}".format(float(self.global_step + start_steps), total_steps, p, self.beta_new))
        # print("lr: {}, alpha: {}, mu: {}".format(self.optimizers().param_groups[0]['lr'], self.alpha, self.mu))

        ## schedule for learning rate
        if self.lr_adaptive == "loss":
            self.adjust_learning_rate_loss(self.optimizers(), self.lr_decay, loss_c_current, loss_c_previous, ">")
        elif self.lr_adaptive is None:
            if self.global_step in [i * start_steps for i in self.lr_steps]:
                self.adjust_learning_rate(self.optimizers(), self.lr_decay)

        if self.lr_adaptive == "dann":
            self.adjust_learning_rate_dann(self.optimizers(), p)

        self.alpha = 2 / (
                    1 + math.exp(-1 * self.current_epoch / self.nb_adapt_epochs)) - 1 if self.alpha < 0 else self.alpha

    def training_step(self, batch, batch_nb):
        """Automatically called by lightning while training a single batch

        Args:
            batch: an item of the dataloader(s) passed with the trainer
            batch_nb: the batch index (which batch is currently being trained)
        Returns:
            The loss(es) calculated after performing an optimiser step
        """

        self._update_batch_epoch_factors(batch_nb)

        loss, task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="T")

        log_metrics = get_aggregated_metrics_from_dict(log_metrics)
        log_metrics.update(get_metrics_from_parameter_dict(self.get_parameters_watch_list(), loss.device))
        log_metrics["T_total_loss"] = loss
        log_metrics["T_adv_loss"] = adv_loss
        log_metrics["T_task_loss"] = task_loss

        for key in log_metrics:
            self.log(key, log_metrics[key])

        return {"loss": loss}

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def multitask_accuracy(self, outputs, labels, topk=(1,)):
        """
        Args:
            outputs: tuple(torch.FloatTensor), each tensor should be of shape
                [batch_size, class_count], class_count can vary on a per task basis, i.e.
                outputs[i].shape[1] can be different to outputs[j].shape[j].
            labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
            topk: tuple(int), compute accuracy at top-k for the values of k specified
                in this parameter.
        Returns:
            tuple(float), same length at topk with the corresponding accuracy@k in.
        """

        max_k = int(np.max(topk))
        task_count = len(outputs)
        batch_size = labels[0].size(0)
        all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor).type_as(labels[0])

        for output, label in zip(outputs, labels):
            _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
            # Flip batch_size, class_count as .view doesn't work on non-contiguous
            max_k_idx = max_k_idx.t()
            correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
            all_correct.add_(correct_for_task)

        accuracies = []
        for k in topk:
            all_tasks_correct = torch.ge(all_correct[:k].float().sum(0), task_count)
            accuracy_at_k = float(all_tasks_correct.float().sum(0) * 100.0 / batch_size)
            accuracies.append(accuracy_at_k)
        return tuple(accuracies)

    # def validation_step(self, batch, batch_idx):
    #     """Automatically called by lightning while validating a single batch
    #
    #     Args:
    #         batch: an item of the dataloader(s) passed with the trainer
    #         batch_idx: the batch index (which batch is currently being validated)
    #     Returns:
    #         The loss(es) calculated
    #     """
    #
    #     # (val_data, val_label, _) = batch
    #     (val_data, val_label) = batch
    #     i = batch_idx
    #
    #     val_size_ori = val_data.size()  # original shape
    #     val_size_ori = val_data.shape  # original shape
    #     batch_val_ori = val_size_ori[0]
    #
    #     # add dummy tensors to keep the same batch size for each epoch (for the last epoch)
    #     if batch_val_ori < self.batch_size[0]:
    #         val_data_dummy = torch.zeros(self.batch_size[0] - batch_val_ori, val_size_ori[1], val_size_ori[2]).type_as(
    #             val_data
    #         )
    #         val_data = torch.cat((val_data, val_data_dummy))
    #
    #     # add dummy tensors to make sure batch size can be divided by gpu #
    #     gpu_count = 1
    #     if val_data.size(0) % gpu_count != 0:
    #         val_data_dummy = torch.zeros(
    #             gpu_count - val_data.size(0) % gpu_count, val_data.size(1), val_data.size(2)
    #         ).type_as(val_data)
    #         val_data = torch.cat((val_data, val_data_dummy))
    #
    #     val_label_verb = val_label[0]
    #     val_label_noun = val_label[1]
    #     with torch.no_grad():
    #
    #         if self.baseline_type == "frame":
    #             val_label_verb_frame = (
    #                 val_label_verb.unsqueeze(1).repeat(1, self.val_segments).view(-1)
    #             )  # expand the size for all the frames
    #             val_label_noun_frame = (
    #                 val_label_noun.unsqueeze(1).repeat(1, self.val_segments).view(-1)
    #             )  # expand the size for all the frames
    #
    #         # compute output
    #         _, _, _, _, _, attn_val, out_val, out_val_2, pred_domain_val, feat_val = self.forward(
    #             val_data, val_data, self.beta, self.mu, is_train=False, reverse=False
    #         )
    #
    #         # ignore dummy tensors
    #         attn_val, out_val, out_val_2, pred_domain_val, feat_val = removeDummy(
    #             attn_val, out_val, out_val_2, pred_domain_val, feat_val, batch_val_ori
    #         )
    #
    #         # measure accuracy and record loss
    #         label_verb = val_label_verb_frame if self.baseline_type == "frame" else val_label_verb
    #         label_noun = val_label_noun_frame if self.baseline_type == "frame" else val_label_noun
    #
    #         # store the embedding
    #         # if self.tensorboard:
    #         # 	self.feat_val_display = feat_val[1] if self.current_epoch == 0 else torch.cat((self.feat_val_display, feat_val[1]), 0)
    #         # 	self.label_val_verb_display = label_verb if self.current_epoch == 0 else torch.cat((self.label_val_verb_display, label_verb), 0)
    #         # 	self.label_val_noun_display = label_noun if self.current_epoch == 0 else torch.cat((self.label_val_noun_display, label_noun), 0)
    #
    #         pred_verb = out_val[0]
    #         pred_noun = out_val[1]
    #
    #         if self.baseline_type == "tsn":
    #             pred_verb = pred_verb.view(val_label.size(0), -1, self.dict_n_class).mean(
    #                 dim=1
    #             )  # average all the segments (needed when num_segments != val_segments)
    #             pred_noun = pred_noun.view(val_label.size(0), -1, self.dict_n_class).mean(
    #                 dim=1
    #             )  # average all the segments (needed when num_segments != val_segments)
    #
    #         loss_verb = self.criterion(pred_verb, label_verb)
    #         loss_noun = self.criterion(pred_noun, label_noun)
    #
    #         loss = 0.5 * (loss_verb + loss_noun)
    #
    #         prec1_verb, prec5_verb = self.accuracy(pred_verb.data, label_verb, topk=(1, 5))
    #         prec1_noun, prec5_noun = self.accuracy(pred_noun.data, label_noun, topk=(1, 5))
    #         prec1_action, prec5_action = self.multitask_accuracy(
    #             (pred_verb.data, pred_noun.data), (label_verb, label_noun), topk=(1, 5)
    #         )
    #
    #     # measure elapsed time
    #     batch_time = time.time() - self.end_val
    #     self.end_val = time.time()
    #
    #     # ======= return dictionary and loggings ======#
    #
    #     result_dict = {
    #         "batch_time": batch_time,
    #         "loss": loss.item(),
    #         "top1_verb": prec1_verb.item(),
    #         "top5_verb": prec5_verb.item(),
    #         "top1_noun": prec1_noun.item(),
    #         "top5_noun": prec5_noun.item(),
    #         "top1_action": prec1_action,
    #         "top5_action": prec5_action,
    #     }
    #
    #     self.log("Prec@1 Verb", result_dict["top1_verb"], prog_bar=True)
    #     self.log("Prec@1 Noun", result_dict["top1_noun"], prog_bar=True)
    #     self.log("Prec@1 Action", result_dict["top1_action"], prog_bar=True)
    #     self.log("Prec@5 Verb", result_dict["top5_verb"], prog_bar=True)
    #     self.log("Prec@5 Noun", result_dict["top5_noun"], prog_bar=True)
    #     self.log("Prec@5 Action", result_dict["top5_action"], prog_bar=True)
    #     self.log("Loss total", result_dict["loss"], prog_bar=True)
    #
    #     return result_dict

    def validation_step(self, batch, batch_nb):

        loss, task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="V")

        log_metrics = get_aggregated_metrics_from_dict(log_metrics)
        log_metrics.update(get_metrics_from_parameter_dict(self.get_parameters_watch_list(), loss.device))
        log_metrics["val_loss"] = loss
        log_metrics["val_adv_loss"] = adv_loss
        log_metrics["val_task_loss"] = task_loss

        for key in log_metrics:
            self.log(key, log_metrics[key])

        # return {"loss": loss}
        return log_metrics

    def validation_epoch_end(self, validation_step_outputs):
        """Automatically called by lightning after validation ends for an epoch
        Args:
            validation_step_outputs: list of values returned by the validation_step after each batch
        """

        self.end_val = time.time()
        # evaluate on validation set

        if self.labels_available:
            self.losses_val = 0
            self.prec1_val = 0
            self.prec1_verb_val = 0
            self.prec1_noun_val = 0
            self.prec5_val = 0
            self.prec5_verb_val = 0
            self.prec5_noun_val = 0

            count = 0
            for dict in validation_step_outputs:
                count += 1
                self.losses_val += dict["loss"]
                self.prec1_val += dict["top1_action"]
                self.prec1_verb_val += dict["top1_verb"]
                self.prec1_noun_val += dict["top1_noun"]
                self.prec5_val += dict["top5_action"]
                self.prec5_verb_val += dict["top5_verb"]
                self.prec5_noun_val += dict["top5_noun"]

            if not (count == 0):
                self.losses_val /= count
                self.prec1_val /= count
                self.prec1_verb_val /= count
                self.prec1_noun_val /= count
                self.prec5_val /= count
                self.prec5_verb_val /= count
                self.prec5_noun_val /= count

            # remember best prec@1 and save checkpoint
            if self.train_metric == "all":
                prec1 = self.prec1_val
            elif self.train_metric == "noun":
                prec1 = self.prec1_noun_val
            elif self.train_metric == "verb":
                prec1 = self.prec1_verb_val
            else:
                raise Exception("invalid metric to train")

            is_best = prec1 > self.best_prec1
            if is_best:
                line_update = " ==> updating the best accuracy" if is_best else ""
                line_best = "Best score {} vs current score {}".format(self.best_prec1, prec1) + line_update
                # log_info(line_best)
            # val_short_file.write('%.3f\n' % prec1)

            self.best_prec1 = max(prec1, self.best_prec1)

    def test_step(self, batch, batch_idx):
        """Automatically called by lightning while testing/infering on a batch

        Args:
            batch: an item of the dataloader(s) passed with the trainer
            batch_idx: the batch index (which batch is currently being evaulated)
        Returns:
            The loss(es) calculated
        """

        # return self.validation_step(batch, batch_idx)
        loss, task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="Te")

        log_metrics = get_aggregated_metrics_from_dict(log_metrics)
        log_metrics.update(get_metrics_from_parameter_dict(self.get_parameters_watch_list(), loss.device))
        log_metrics["test_loss"] = loss
        return log_metrics

    def add_dummy_data(self, x_rgb, x_flow, x_audio, batch_size):
        if self.rgb:
            current_size = x_rgb.size()
            data_dummy = torch.zeros(batch_size - current_size[0], current_size[1], current_size[2])
            data_dummy = data_dummy.type_as(x_rgb)
            x_rgb = torch.cat((x_rgb, data_dummy))
        if self.flow:
            current_size = x_flow.size()
            data_dummy = torch.zeros(batch_size - current_size[0], current_size[1], current_size[2])
            data_dummy = data_dummy.type_as(x_flow)
            x_flow = torch.cat((x_flow, data_dummy))
        if self.audio:
            current_size = x_audio.size()
            data_dummy = torch.zeros(batch_size - current_size[0], current_size[1], current_size[2])
            data_dummy = data_dummy.type_as(x_audio)
            x_audio = torch.cat((x_audio, data_dummy))
        return x_rgb, x_flow, x_audio

    def remove_dummy(self, y_hat, d_hat_rgb, d_hat_flow, d_hat_audio, d_hat_0_1, batch_size):
        y_hat[0] = y_hat[0][:batch_size]
        y_hat[1] = y_hat[1][:batch_size]
        if self.rgb:
            d_hat_rgb = d_hat_rgb[:batch_size]
        if self.flow:
            d_hat_flow = d_hat_flow[:batch_size]
        if self.audio:
            d_hat_audio = d_hat_audio[:batch_size]
        d_hat_0_1 = [
            [d[0][:batch_size], d[1][:batch_size]] if d is not None and d[0] is not None else [None, None]
            for d in d_hat_0_1
        ]
        return y_hat, d_hat_rgb, d_hat_flow, d_hat_audio, d_hat_0_1

    def removeDummy(self, attn, out_1, out_2, pred_domain, feat, batch_size):
        attn = attn[:batch_size]
        if isinstance(out_1, (list, tuple)):
            out_1 = (out_1[0][:batch_size], out_1[1][:batch_size])
        else:
            out_1 = out_1[:batch_size]
        out_2 = out_2[:batch_size]
        pred_domain = [pred[:batch_size] for pred in pred_domain]
        feat = [f[:batch_size] for f in feat]

        return attn, out_1, out_2, pred_domain, feat
