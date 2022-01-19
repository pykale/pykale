# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Domain adaptation systems (pipelines) for video data, e.g., for action recognition.
Most are inherited from kale.pipeline.domain_adapter.
"""

import torch

import kale.predict.losses as losses
from kale.loaddata.video_access import get_image_modality
from kale.pipeline.domain_adapter import (
    BaseMMDLike,
    CDANTrainer,
    DANNTrainer,
    get_aggregated_metrics_from_dict,
    get_metrics_from_parameter_dict,
    GradReverse,
    Method,
    set_requires_grad,
    WDGRLTrainer,
)


def create_mmd_based_video(method: Method, dataset, image_modality, feature_extractor, task_classifier, **train_params):
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
            **train_params,
        )
    if method is Method.JAN:
        return JANTrainerVideo(
            dataset=dataset,
            image_modality=image_modality,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            method=method,
            kernel_mul=[2.0, 2.0],
            kernel_num=[5, 1],
            **train_params,
        )


def create_dann_like_video(
    method: Method, dataset, image_modality, feature_extractor, task_classifier, critic, **train_params
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
            **train_params,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")


class BaseMMDLikeVideo(BaseMMDLike):
    """Common API for MME-based domain adaptation on video data: DAN, JAN"""

    def __init__(
        self, dataset, image_modality, feature_extractor, task_classifier, kernel_mul=2.0, kernel_num=5, **base_params,
    ):
        super().__init__(dataset, feature_extractor, task_classifier, kernel_mul, kernel_num, **base_params)
        self.image_modality = image_modality
        self.rgb_feat = self.feat["rgb"]
        self.flow_feat = self.feat["flow"]

    def forward(self, x):
        if self.feat is not None:
            if self.image_modality in ["rgb", "flow"]:
                if self.rgb_feat is not None:
                    x = self.rgb_feat(x)
                else:
                    x = self.flow_feat(x)
                x = x.view(x.size(0), -1)
                class_output = self.classifier(x)
                return x, class_output

            elif self.image_modality == "joint":
                x_rgb = self.rgb_feat(x["rgb"])
                x_flow = self.flow_feat(x["flow"])
                x_rgb = x_rgb.view(x_rgb.size(0), -1)
                x_flow = x_flow.view(x_flow.size(0), -1)
                x = torch.cat((x_rgb, x_flow), dim=1)
                class_output = self.classifier(x)
                return [x_rgb, x_flow], class_output

    def compute_loss(self, batch, split_name="valid"):
        # _s refers to source, _tu refers to unlabeled target
        if self.image_modality == "joint" and len(batch) == 4:
            (x_s_rgb, y_s), (x_s_flow, y_s_flow), (x_tu_rgb, y_tu), (x_tu_flow, y_tu_flow) = batch
            [phi_s_rgb, phi_s_flow], y_hat = self.forward({"rgb": x_s_rgb, "flow": x_s_flow})
            [phi_t_rgb, phi_t_flow], y_t_hat = self.forward({"rgb": x_tu_rgb, "flow": x_tu_flow})
            mmd_rgb = self._compute_mmd(phi_s_rgb, phi_t_rgb, y_hat, y_t_hat)
            mmd_flow = self._compute_mmd(phi_s_flow, phi_t_flow, y_hat, y_t_hat)
            mmd = mmd_rgb + mmd_flow
        elif self.image_modality in ["rgb", "flow"] and len(batch) == 2:
            (x_s, y_s), (x_tu, y_tu) = batch
            phi_s, y_hat = self.forward(x_s)
            phi_t, y_t_hat = self.forward(x_tu)
            mmd = self._compute_mmd(phi_s, phi_t, y_hat, y_t_hat)
        else:
            raise NotImplementedError("Batch len is {}. Check the Dataloader.".format(len(batch)))

        # Uncomment when checking whether rgb & flow labels are equal.
        # print('rgb_s:{}, flow_s:{}, rgb_f:{}, flow_f:{}'.format(y_s, y_s_flow, y_tu, y_tu_flow))
        # print('equal: {}/{}'.format(torch.all(torch.eq(y_s, y_s_flow)), torch.all(torch.eq(y_tu, y_tu_flow))))

        # ok is abbreviation for (all) correct
        loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)
        task_loss = loss_cls
        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": mmd,
        }
        return task_loss, mmd, log_metrics


class DANTrainerVideo(BaseMMDLikeVideo):
    """This is an implementation of DAN for video data."""

    def __init__(self, dataset, image_modality, feature_extractor, task_classifier, **base_params):
        super().__init__(dataset, image_modality, feature_extractor, task_classifier, **base_params)

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):
        batch_size = int(phi_s.size()[0])
        kernels = losses.gaussian_kernel(phi_s, phi_t, kernel_mul=self._kernel_mul, kernel_num=self._kernel_num,)
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


class DANNTrainerVideo(DANNTrainer):
    """This is an implementation of DANN for video data."""

    def __init__(
        self, dataset, image_modality, feature_extractor, task_classifier, critic, method, **base_params,
    ):
        super(DANNTrainerVideo, self).__init__(
            dataset, feature_extractor, task_classifier, critic, method, **base_params
        )
        self.image_modality = image_modality
        self.rgb, self.flow = get_image_modality(self.image_modality)
        self.rgb_feat = self.feat["rgb"]
        self.flow_feat = self.feat["flow"]

    def forward(self, x):
        if self.feat is not None:
            x_rgb = x_flow = adversarial_output_rgb = adversarial_output_flow = None

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

            if self.rgb:
                if self.flow:  # For joint input
                    x = torch.cat((x_rgb, x_flow), dim=1)
                else:  # For rgb input
                    x = x_rgb
            else:  # For flow input
                x = x_flow
            class_output = self.classifier(x)

            return [x_rgb, x_flow], class_output, [adversarial_output_rgb, adversarial_output_flow]

    def compute_loss(self, batch, split_name="valid"):
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

        if self.rgb:
            loss_dmn_src_rgb, dok_src_rgb = losses.cross_entropy_logits(d_hat_rgb, torch.zeros(batch_size))
            loss_dmn_tgt_rgb, dok_tgt_rgb = losses.cross_entropy_logits(d_t_hat_rgb, torch.ones(batch_size))
        if self.flow:
            loss_dmn_src_flow, dok_src_flow = losses.cross_entropy_logits(d_hat_flow, torch.zeros(batch_size))
            loss_dmn_tgt_flow, dok_tgt_flow = losses.cross_entropy_logits(d_t_hat_flow, torch.ones(batch_size))

        if self.rgb and self.flow:  # For joint input
            loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_flow
            loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_flow
            dok = torch.cat((dok_src_rgb, dok_src_flow, dok_tgt_rgb, dok_tgt_flow))
            dok_src = torch.cat((dok_src_rgb, dok_src_flow))
            dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_flow))
        else:
            if self.rgb:  # For rgb input
                d_hat = d_hat_rgb
                d_t_hat = d_t_hat_rgb
            else:  # For flow input
                d_hat = d_hat_flow
                d_t_hat = d_t_hat_flow

            # ok is abbreviation for (all) correct, dok refers to domain correct
            loss_dmn_src, dok_src = losses.cross_entropy_logits(d_hat, torch.zeros(batch_size))
            loss_dmn_tgt, dok_tgt = losses.cross_entropy_logits(d_t_hat, torch.ones(batch_size))
            dok = torch.cat((dok_src, dok_tgt))

        loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)
        adv_loss = loss_dmn_src + loss_dmn_tgt  # adv_loss = src + tgt
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": dok,
            f"{split_name}_source_domain_acc": dok_src,
            f"{split_name}_target_domain_acc": dok_tgt,
        }

        return task_loss, adv_loss, log_metrics

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


class CDANTrainerVideo(CDANTrainer):
    """This is an implementation of CDAN for video data."""

    def __init__(
        self,
        dataset,
        image_modality,
        feature_extractor,
        task_classifier,
        critic,
        use_entropy=False,
        use_random=False,
        random_dim=1024,
        **base_params,
    ):
        super(CDANTrainerVideo, self).__init__(
            dataset, feature_extractor, task_classifier, critic, use_entropy, use_random, random_dim, **base_params
        )
        self.image_modality = image_modality
        self.rgb, self.flow = get_image_modality(image_modality)
        self.rgb_feat = self.feat["rgb"]
        self.flow_feat = self.feat["flow"]

    def forward(self, x):
        if self.feat is not None:
            x_rgb = x_flow = adversarial_output_rgb = adversarial_output_flow = None

            # For joint input, both two ifs are used
            if self.rgb:
                x_rgb = self.rgb_feat(x["rgb"])
                x_rgb = x_rgb.view(x_rgb.size(0), -1)
                reverse_feature_rgb = GradReverse.apply(x_rgb, self.alpha)
            if self.flow:
                x_flow = self.flow_feat(x["flow"])
                x_flow = x_flow.view(x_flow.size(0), -1)
                reverse_feature_flow = GradReverse.apply(x_flow, self.alpha)

            if self.rgb:
                if self.flow:  # For joint input
                    x = torch.cat((x_rgb, x_flow), dim=1)
                else:  # For rgb input
                    x = x_rgb
            else:  # For flow input
                x = x_flow
            class_output = self.classifier(x)
            softmax_output = torch.nn.Softmax(dim=1)(class_output)
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
            return [x_rgb, x_flow], class_output, [adversarial_output_rgb, adversarial_output_flow]

    def compute_loss(self, batch, split_name="valid"):
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

        if self.entropy:
            e_s = self._compute_entropy_weights(y_hat)
            e_t = self._compute_entropy_weights(y_t_hat)
            source_weight = e_s / torch.sum(e_s)
            target_weight = e_t / torch.sum(e_t)
        else:
            source_weight = None
            target_weight = None

        if self.rgb:
            loss_dmn_src_rgb, dok_src_rgb = losses.cross_entropy_logits(
                d_hat_rgb, torch.zeros(batch_size), source_weight
            )
            loss_dmn_tgt_rgb, dok_tgt_rgb = losses.cross_entropy_logits(
                d_t_hat_rgb, torch.ones(batch_size), target_weight
            )

        if self.flow:
            loss_dmn_src_flow, dok_src_flow = losses.cross_entropy_logits(
                d_hat_flow, torch.zeros(batch_size), source_weight
            )
            loss_dmn_tgt_flow, dok_tgt_flow = losses.cross_entropy_logits(
                d_t_hat_flow, torch.ones(batch_size), target_weight
            )

        # ok is abbreviation for (all) correct, dok refers to domain correct
        if self.rgb and self.flow:  # For joint input
            loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_flow
            loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_flow
            dok = torch.cat((dok_src_rgb, dok_src_flow, dok_tgt_rgb, dok_tgt_flow))
            dok_src = torch.cat((dok_src_rgb, dok_src_flow))
            dok_tgt = torch.cat((dok_tgt_rgb, dok_tgt_flow))
        else:
            if self.rgb:  # For rgb input
                d_hat = d_hat_rgb
                d_t_hat = d_t_hat_rgb
            else:  # For flow input
                d_hat = d_hat_flow
                d_t_hat = d_t_hat_flow

            loss_dmn_src, dok_src = losses.cross_entropy_logits(d_hat, torch.zeros(batch_size))
            loss_dmn_tgt, dok_tgt = losses.cross_entropy_logits(d_t_hat, torch.ones(batch_size))
            dok = torch.cat((dok_src, dok_tgt))

        loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)
        adv_loss = loss_dmn_src + loss_dmn_tgt  # adv_loss = src + tgt
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": dok,
            f"{split_name}_source_domain_acc": dok_src,
            f"{split_name}_target_domain_acc": dok_tgt,
        }

        return task_loss, adv_loss, log_metrics


class WDGRLTrainerVideo(WDGRLTrainer):
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
        self.rgb, self.flow = get_image_modality(self.image_modality)
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

    def compute_loss(self, batch, split_name="valid"):
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
                # gp_x refers to gradient penelty for the input with the modality x.
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
