import pytorch_lightning as pl

from kale.pipeline import domain_adapter


class ModelTestHelper:
    @staticmethod
    def test_model(model, train_params, **kwargs):
        assert isinstance(model, domain_adapter.BaseAdaptTrainer)
        # training process
        trainer = pl.Trainer(
            default_root_dir="tests/outputs",
            min_epochs=train_params["nb_init_epochs"],
            max_epochs=train_params["nb_adapt_epochs"],
            devices=1,
            **kwargs,
        )
        trainer.fit(model)
        trainer.test()
        metric_values = trainer.callback_metrics
        assert isinstance(metric_values, dict)
