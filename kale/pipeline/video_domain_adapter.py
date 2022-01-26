# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Domain adaptation systems (pipelines) for video data, e.g., for action recognition.
Most are inherited from kale.pipeline.domain_adapter.
"""

import torch

import kale.predict.losses as losses
from kale.loaddata.video_access import get_class_type, get_image_modality
from kale.pipeline.domain_adapter import (
    BaseAdaptTrainer,
    BaseMMDLike,
    CDANTrainer,
    DANNTrainer,
    get_aggregated_metrics,
    get_aggregated_metrics_from_dict,
    get_metrics_from_parameter_dict,
    GradReverse,
    Method,
    set_requires_grad,
    WDGRLTrainer,
)

# from kale.utils.logger import save_results_to_json


def create_mmd_based_video(
    method: Method, dataset, image_modality, feature_extractor, task_classifier, class_type, **train_params
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
            class_type=class_type,
            kernel_mul=[2.0, 2.0],
            kernel_num=[5, 1],
            **train_params,
        )


def create_dann_like_video(
    method: Method, dataset, image_modality, feature_extractor, task_classifier, critic, class_type, **train_params
):
    """DANN-based deep learning methods for domain adaptation on video data: DANN, CDAN, CDAN+E"""

    # Uncomment for later work.
    # Set up a new create_fewshot_trainer for video data based on original one in `domain_adapter.py`

    # if dataset.is_semi_supervised():
    #     return create_fewshot_trainer_video(
    #         method, dataset, feature_extractor, task_classifier, critic, **train_params
    #     )

    if method.is_dann_method():
        alpha = 0.0 if method is Method.Source else 1.0
        return DANNTrainerVideo(
            alpha=alpha,
            image_modality=image_modality,
            dataset=dataset,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
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
            class_type=class_type,
            use_entropy=method is Method.CDAN_E,
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
            class_type=class_type,
            **train_params,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")


class BaseAdaptTrainerVideo(BaseAdaptTrainer):
    """Base class for all domain adaptation architectures on videos. Inherited from BaseAdaptTrainer."""

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
        self._target_batch_size = target_batch_size
        return dataloader

    def training_step(self, batch, batch_nb):
        self._update_batch_epoch_factors(batch_nb)

        task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="train")
        if self.current_epoch < self._init_epochs:
            loss = task_loss
        else:
            loss = task_loss + self.lamb_da * adv_loss

        log_metrics = get_aggregated_metrics_from_dict(log_metrics)
        log_metrics.update(get_metrics_from_parameter_dict(self.get_parameters_watch_list(), loss.device))
        log_metrics["train_total_loss"] = loss
        log_metrics["train_adv_loss"] = adv_loss
        log_metrics["train_task_loss"] = task_loss

        for key in log_metrics:
            self.log(key, log_metrics[key])

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        metrics_to_log = self.create_metrics_log("valid")
        return self._validation_epoch_end(outputs, metrics_to_log)

    def test_epoch_end(self, outputs):
        metrics_at_test = self.create_metrics_log("test")

        # Uncomment to save output to json file for EPIC UDA 2021 challenge.(3/3)
        # save_results_to_json(
        #     self.y_hat, self.y_t_hat, self.s_id, self.tu_id, self.y_hat_noun, self.y_t_hat_noun, self.verb, self.noun
        # )

        log_dict = get_aggregated_metrics(metrics_at_test, outputs)

        for key in log_dict:
            self.log(key, log_dict[key], prog_bar=True)

    def concatenate(self, x_rgb, x_flow, x_audio):
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

    def create_metrics_log(self, split_name):
        if self.verb and not self.noun:
            metrics_to_log = (
                "{}_loss".format(split_name),
                "{}_task_loss".format(split_name),
                "{}_adv_loss".format(split_name),
                # "{}_source_acc".format(split_name),
                "{}_source_top1_acc".format(split_name),
                "{}_source_top5_acc".format(split_name),
                # "{}_target_acc".format(split_name),
                "{}_target_top1_acc".format(split_name),
                "{}_target_top5_acc".format(split_name),
                "{}_source_domain_acc".format(split_name),
                "{}_target_domain_acc".format(split_name),
                "{}_domain_acc".format(split_name),
            )
            if self.method.is_mmd_method():
                metrics_to_log = metrics_to_log[:-3] + ("{}_mmd".format(split_name),)
        elif self.verb and self.noun:
            metrics_to_log = (
                "{}_loss".format(split_name),
                "{}_task_loss".format(split_name),
                "{}_adv_loss".format(split_name),
                # "{}_verb_source_acc".format(split_name),
                "{}_verb_source_top1_acc".format(split_name),
                "{}_verb_source_top5_acc".format(split_name),
                # "{}_noun_source_acc".format(split_name),
                "{}_noun_source_top1_acc".format(split_name),
                "{}_noun_source_top5_acc".format(split_name),
                # "{}_verb_target_acc".format(split_name),
                "{}_verb_target_top1_acc".format(split_name),
                "{}_verb_target_top5_acc".format(split_name),
                # "{}_noun_target_acc".format(split_name),
                "{}_noun_target_top1_acc".format(split_name),
                "{}_noun_target_top5_acc".format(split_name),
                "{}_action_source_top1_acc".format(split_name),
                "{}_action_source_top5_acc".format(split_name),
                "{}_action_target_top1_acc".format(split_name),
                "{}_action_target_top5_acc".format(split_name),
                "{}_domain_acc".format(split_name),
            )
            if self.method.is_mmd_method():
                metrics_to_log = metrics_to_log[:-1] + ("{}_mmd".format(split_name),)
        if split_name == "test":
            metrics_to_log = metrics_to_log[:1] + metrics_to_log[3:]
        return metrics_to_log

    def get_loss_log_metrics(self, split_name, y_hat, y_t_hat, y_s, y_tu, dok):
        """Get the loss, top-k accuracy and metrics for a given split."""

        if self.verb and not self.noun:
            loss_cls, ok_src = losses.cross_entropy_logits(y_hat[0], y_s[0])
            _, ok_tgt = losses.cross_entropy_logits(y_t_hat[0], y_tu[0])
            prec1_src, prec5_src = losses.topk_accuracy(y_hat[0], y_s[0], topk=(1, 5))
            prec1_tgt, prec5_tgt = losses.topk_accuracy(y_t_hat[0], y_tu[0], topk=(1, 5))
            task_loss = loss_cls

            log_metrics = {
                # f"{split_name}_source_acc": ok_src,
                # f"{split_name}_target_acc": ok_tgt,
                f"{split_name}_source_top1_acc": prec1_src,
                f"{split_name}_source_top5_acc": prec5_src,
                f"{split_name}_target_top1_acc": prec1_tgt,
                f"{split_name}_target_top5_acc": prec5_tgt,
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
            }

        if self.method.is_mmd_method():
            log_metrics.update({f"{split_name}_mmd": dok})
        else:
            log_metrics.update({f"{split_name}_domain_acc": dok})
        return task_loss, log_metrics


class BaseMMDLikeVideo(BaseAdaptTrainerVideo, BaseMMDLike):
    """Common API for MME-based domain adaptation on video data: DAN, JAN"""

    def __init__(
        self,
        dataset,
        image_modality,
        feature_extractor,
        task_classifier,
        class_type,
        kernel_mul=2.0,
        kernel_num=5,
        **base_params,
    ):
        super().__init__(dataset, feature_extractor, task_classifier, kernel_mul, kernel_num, **base_params)
        self.image_modality = image_modality
        self.rgb, self.flow, self.audio = get_image_modality(self.image_modality)
        self.class_type = class_type
        self.verb, self.noun = get_class_type(self.class_type)
        self.rgb_feat = self.feat["rgb"]
        self.flow_feat = self.feat["flow"]
        self.audio_feat = self.feat["audio"]

    def forward(self, x):
        if self.feat is not None:
            x_rgb = x_flow = x_audio = None

            if self.rgb:
                x_rgb = self.rgb_feat(x["rgb"])
                x_rgb = x_rgb.view(x_rgb.size(0), -1)
            if self.flow:
                x_flow = self.flow_feat(x["flow"])
                x_flow = x_flow.view(x_flow.size(0), -1)
            if self.audio:
                x_audio = self.audio_feat(x["audio"])
                x_audio = x_audio.view(x_audio.size(0), -1)

            x = self.concatenate(x_rgb, x_flow, x_audio)
            class_output = self.classifier(x)
            return [x_rgb, x_flow, x_audio], class_output

    def compute_loss(self, batch, split_name="valid"):
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

        # _s refers to source, _tu refers to unlabeled target
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
        # log_metrics.update({f"{split_name}_source_domain_acc": mmd, f"{split_name}_target_domain_acc": mmd})
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


class DANNTrainerVideo(BaseAdaptTrainerVideo, DANNTrainer):
    """This is an implementation of DANN for video data."""

    def __init__(
        self, dataset, image_modality, feature_extractor, task_classifier, critic, method, class_type, **base_params,
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

            x = self.concatenate(x_rgb, x_flow, x_audio)

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
        # if split_name == "test":
        #     self.y_hat.extend(y_hat[0].tolist())
        #     self.y_hat_noun.extend(y_hat[1].tolist())
        #     self.y_t_hat.extend(y_t_hat[0].tolist())
        #     self.y_t_hat_noun.extend(y_t_hat[1].tolist())
        #     self.s_id.extend(s_id)
        #     self.tu_id.extend(tu_id)

        return task_loss, adv_loss, log_metrics


class CDANTrainerVideo(BaseAdaptTrainerVideo, CDANTrainer):
    """This is an implementation of CDAN for video data."""

    def __init__(
        self,
        dataset,
        image_modality,
        feature_extractor,
        task_classifier,
        critic,
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

            x = self.concatenate(x_rgb, x_flow, x_audio)

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


class WDGRLTrainerVideo(BaseAdaptTrainerVideo, WDGRLTrainer):
    """This is an implementation of WDGRL for video data."""

    def __init__(
        self,
        dataset,
        image_modality,
        feature_extractor,
        task_classifier,
        critic,
        class_type,
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
        self.class_type = class_type
        self.verb, self.noun = get_class_type(self.class_type)
        self.rgb_feat = self.feat["rgb"]
        self.flow_feat = self.feat["flow"]
        self.audio_feat = self.feat["audio"]

    def forward(self, x):
        if self.feat is not None:
            x_rgb = x_flow = x_audio = None
            adversarial_output_rgb = adversarial_output_flow = adversarial_output_audio = None

            # For joint input, both two ifs are used
            if self.rgb:
                x_rgb = self.rgb_feat(x["rgb"])
                x_rgb = x_rgb.view(x_rgb.size(0), -1)
                adversarial_output_rgb = self.domain_classifier(x_rgb)
            if self.flow:
                x_flow = self.flow_feat(x["flow"])
                x_flow = x_flow.view(x_flow.size(0), -1)
                adversarial_output_flow = self.domain_classifier(x_flow)
            if self.audio:
                x_audio = self.audio_feat(x["audio"])
                x_audio = x_audio.view(x_audio.size(0), -1)
                adversarial_output_audio = self.domain_classifier(x_audio)

            x = self.concatenate(x_rgb, x_flow, x_audio)

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

        if self.rgb:
            if self.flow:
                if self.audio:  # For all inputs
                    dok = torch.cat(
                        (dok_src_rgb, dok_src_flow, dok_src_audio, dok_tgt_rgb, dok_tgt_flow, dok_tgt_audio)
                    )
                    dok_src = torch.cat((dok_src_rgb, dok_src_flow, dok_src_audio))
                    dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_flow, dok_tgt_audio))
                    wasserstein_distance_rgb = d_hat_rgb.mean() - (1 + self._beta_ratio) * d_t_hat_rgb.mean()
                    wasserstein_distance_flow = d_hat_flow.mean() - (1 + self._beta_ratio) * d_t_hat_flow.mean()
                    wasserstein_distance_audio = d_hat_audio.mean() - (1 + self._beta_ratio) * d_t_hat_audio.mean()
                    wasserstein_distance = (
                        wasserstein_distance_rgb + wasserstein_distance_flow + wasserstein_distance_audio
                    ) / 3
                else:  # For joint(rgb+flow) input
                    dok = torch.cat((dok_src_rgb, dok_src_flow, dok_tgt_rgb, dok_tgt_flow))
                    dok_src = torch.cat((dok_src_rgb, dok_src_flow))
                    dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_flow))
                    wasserstein_distance_rgb = d_hat_rgb.mean() - (1 + self._beta_ratio) * d_t_hat_rgb.mean()
                    wasserstein_distance_flow = d_hat_flow.mean() - (1 + self._beta_ratio) * d_t_hat_flow.mean()
                    wasserstein_distance = (wasserstein_distance_rgb + wasserstein_distance_flow) / 2
            else:
                if self.audio:  # For rgb+audio input
                    dok = torch.cat((dok_src_rgb, dok_src_audio, dok_tgt_rgb, dok_tgt_audio))
                    dok_src = torch.cat((dok_src_rgb, dok_src_audio))
                    dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_audio))
                    wasserstein_distance_rgb = d_hat_rgb.mean() - (1 + self._beta_ratio) * d_t_hat_rgb.mean()
                    wasserstein_distance_audio = d_hat_audio.mean() - (1 + self._beta_ratio) * d_t_hat_audio.mean()
                    wasserstein_distance = (wasserstein_distance_rgb + wasserstein_distance_audio) / 2
                else:  # For rgb input
                    d_hat = d_hat_rgb
                    d_t_hat = d_t_hat_rgb
                    dok_src = dok_src_rgb
                    dok_tgt = dok_tgt_rgb
                    wasserstein_distance = d_hat.mean() - (1 + self._beta_ratio) * d_t_hat.mean()
                    dok = torch.cat((dok_src, dok_tgt))
        else:
            if self.flow:
                if self.audio:  # For flow+audio input
                    dok = torch.cat((dok_src_audio, dok_src_flow, dok_tgt_audio, dok_tgt_flow))
                    dok_src = torch.cat((dok_src_audio, dok_src_flow))
                    dok_tgt = torch.cat((dok_tgt_audio, dok_tgt_flow))
                    wasserstein_distance_audio = d_hat_audio.mean() - (1 + self._beta_ratio) * d_t_hat_audio.mean()
                    wasserstein_distance_flow = d_hat_flow.mean() - (1 + self._beta_ratio) * d_t_hat_flow.mean()
                    wasserstein_distance = (wasserstein_distance_audio + wasserstein_distance_flow) / 2
                else:  # For flow input
                    d_hat = d_hat_flow
                    d_t_hat = d_t_hat_flow
                    dok_src = dok_src_flow
                    dok_tgt = dok_tgt_flow
                    wasserstein_distance = d_hat.mean() - (1 + self._beta_ratio) * d_t_hat.mean()
                    dok = torch.cat((dok_src, dok_tgt))
            else:  # For audio input
                d_hat = d_hat_audio
                d_t_hat = d_t_hat_audio
                dok_src = dok_src_audio
                dok_tgt = dok_tgt_audio
                wasserstein_distance = d_hat.mean() - (1 + self._beta_ratio) * d_t_hat.mean()
                dok = torch.cat((dok_src, dok_tgt))

        task_loss, log_metrics = self.get_loss_log_metrics(split_name, y_hat, y_t_hat, y_s, y_tu, dok)
        adv_loss = wasserstein_distance
        log_metrics.update({f"{split_name}_source_domain_acc": dok_src, f"{split_name}_target_domain_acc": dok_tgt})

        return task_loss, adv_loss, log_metrics

    def configure_optimizers(self):
        nets = [self.classifier]
        if self.rgb:
            nets.append(self.rgb_feat)
        if self.flow:
            nets.append(self.flow_feat)
        if self.audio:
            nets.append(self.audio_feat)

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

        if self.image_modality in ["rgb", "flow", "audio"]:
            if self.rgb:
                set_requires_grad(self.rgb_feat, requires_grad=False)
                (x_s, y_s), (x_tu, _) = batch
                with torch.no_grad():
                    h_s = self.rgb_feat(x_s).data.view(x_s.shape[0], -1)
                    h_t = self.rgb_feat(x_tu).data.view(x_tu.shape[0], -1)
            if self.flow:
                set_requires_grad(self.flow_feat, requires_grad=False)
                (x_s, y_s), (x_tu, _) = batch
                with torch.no_grad():
                    h_s = self.flow_feat(x_s).data.view(x_s.shape[0], -1)
                    h_t = self.flow_feat(x_tu).data.view(x_tu.shape[0], -1)
            if self.audio:
                set_requires_grad(self.audio_feat, requires_grad=False)
                (x_s, y_s), (x_tu, _) = batch
                with torch.no_grad():
                    h_s = self.audio_feat(x_s).data.view(x_s.shape[0], -1)
                    h_t = self.audio_feat(x_tu).data.view(x_tu.shape[0], -1)

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

            if self.rgb:
                set_requires_grad(self.rgb_feat, requires_grad=True)
            if self.flow:
                set_requires_grad(self.flow_feat, requires_grad=True)
            if self.audio:
                set_requires_grad(self.audio_feat, requires_grad=True)
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
                # h_s = torch.cat((h_s_rgb, h_s_flow), dim=1)
                # h_t = torch.cat((h_t_rgb, h_t_flow), dim=1)

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

        elif self.image_modality == "all":
            set_requires_grad(self.rgb_feat, requires_grad=False)
            set_requires_grad(self.flow_feat, requires_grad=False)
            set_requires_grad(self.audio_feat, requires_grad=False)
            (x_s_rgb, y_s), (x_s_flow, _), (x_s_audio, _), (x_tu_rgb, _), (x_tu_flow, _), (x_tu_audio, _) = batch
            with torch.no_grad():
                h_s_rgb = self.rgb_feat(x_s_rgb).data.view(x_s_rgb.shape[0], -1)
                h_t_rgb = self.rgb_feat(x_tu_rgb).data.view(x_tu_rgb.shape[0], -1)
                h_s_flow = self.flow_feat(x_s_flow).data.view(x_s_flow.shape[0], -1)
                h_t_flow = self.flow_feat(x_tu_flow).data.view(x_tu_flow.shape[0], -1)
                h_s_audio = self.audio_feat(x_s_audio).data.view(x_s_audio.shape[0], -1)
                h_t_audio = self.audio_feat(x_tu_audio).data.view(x_tu_audio.shape[0], -1)
                # h_s = torch.cat((h_s_rgb, h_s_flow), dim=1)
                # h_t = torch.cat((h_t_rgb, h_t_flow), dim=1)

            # Need to improve to process rgb and flow separately in the future.
            for _ in range(self._k_critic):
                # gp_x refers to gradient penelty for the input with the image_modality x.
                gp_rgb = losses.gradient_penalty(self.domain_classifier, h_s_rgb, h_t_rgb)
                gp_flow = losses.gradient_penalty(self.domain_classifier, h_s_flow, h_t_flow)
                gp_audio = losses.gradient_penalty(self.domain_classifier, h_s_audio, h_t_audio)

                critic_s_rgb = self.domain_classifier(h_s_rgb)
                critic_s_flow = self.domain_classifier(h_s_flow)
                critic_s_audio = self.domain_classifier(h_s_audio)
                critic_t_rgb = self.domain_classifier(h_t_rgb)
                critic_t_flow = self.domain_classifier(h_t_flow)
                critic_t_audio = self.domain_classifier(h_t_audio)
                wasserstein_distance_rgb = critic_s_rgb.mean() - (1 + self._beta_ratio) * critic_t_rgb.mean()
                wasserstein_distance_flow = critic_s_flow.mean() - (1 + self._beta_ratio) * critic_t_flow.mean()
                wasserstein_distance_audio = critic_s_audio.mean() - (1 + self._beta_ratio) * critic_t_audio.mean()

                critic_cost = (
                    -wasserstein_distance_rgb
                    + -wasserstein_distance_flow
                    + -wasserstein_distance_audio
                    + self._gamma * gp_rgb
                    + self._gamma * gp_flow
                    + self._gamma * gp_audio
                ) * 0.5

                self.critic_opt.zero_grad()
                critic_cost.backward()
                self.critic_opt.step()
                if self.critic_sched:
                    self.critic_sched.step()

            set_requires_grad(self.rgb_feat, requires_grad=True)
            set_requires_grad(self.flow_feat, requires_grad=True)
            set_requires_grad(self.audio_feat, requires_grad=True)
            set_requires_grad(self.domain_classifier, requires_grad=False)
