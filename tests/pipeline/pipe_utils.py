import pytorch_lightning as pl
from kale.pipeline import domain_adapter


def test_model(model, train_params):
    assert isinstance(model, domain_adapter.BaseAdaptTrainer)
    # training process
    trainer = pl.Trainer(min_epochs=train_params["nb_init_epochs"], max_epochs=train_params["nb_adapt_epochs"], gpus=0)
    trainer.fit(model)
    trainer.test()
    metric_values = trainer.callback_metrics
    assert isinstance(metric_values, dict)
