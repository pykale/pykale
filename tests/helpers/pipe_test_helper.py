import pytorch_lightning as pl

from kale.pipeline import domain_adapter, video_domain_adapter
from kale.predict.class_domain_nets import DomainNetVideo


class ModelTestHelper:
    @staticmethod
    def test_model(model, train_params, **kwargs):
        assert isinstance(model, domain_adapter.BaseAdaptTrainer)
        # training process
        trainer = pl.Trainer(
            min_epochs=train_params["nb_init_epochs"], max_epochs=train_params["nb_adapt_epochs"], gpus=0, **kwargs
        )
        trainer.fit(model)
        trainer.test()
        metric_values = trainer.callback_metrics
        assert isinstance(metric_values, dict)


class DASetupHelper:
    @staticmethod
    def setup_da(da_method, dataset, feature_network, classifier_network, class_type, train_params, domain_feature_dim,
                 dict_num_classes, cfg):
        method_params = {}
        method = domain_adapter.Method(da_method)

        # setup DA method
        if method.is_mmd_method():
            model = video_domain_adapter.create_mmd_based_video(
                method=method,
                dataset=dataset,
                image_modality=cfg.DATASET.IMAGE_MODALITY,
                feature_extractor=feature_network,
                task_classifier=classifier_network,
                class_type=class_type,
                **method_params,
                **train_params,
            )
        else:
            critic_input_size = domain_feature_dim
            # setup critic network
            if method.is_cdan_method():
                if cfg.DAN.USERANDOM:
                    critic_input_size = 10
                else:
                    critic_input_size = domain_feature_dim * dict_num_classes["verb"]
            critic_network = DomainNetVideo(input_size=critic_input_size)

            if da_method == "CDAN":
                method_params["use_random"] = cfg.DAN.USERANDOM

            model = video_domain_adapter.create_dann_like_video(
                method=method,
                dataset=dataset,
                image_modality=cfg.DATASET.IMAGE_MODALITY,
                feature_extractor=feature_network,
                task_classifier=classifier_network,
                critic=critic_network,
                class_type=class_type,
                **method_params,
                **train_params,
            )
        return model
