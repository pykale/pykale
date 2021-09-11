def eval_pipeline(trainer, model, method, results, test_csv_file, seed):
    trainer.fit(model)
    results.update(
        is_validation=True, method_name=method, seed=seed, metric_values=trainer.callback_metrics,
    )
    # test scores
    trainer.test()
    results.update(
        is_validation=False, method_name=method, seed=seed, metric_values=trainer.callback_metrics,
    )
    results.to_csv(test_csv_file)
    results.print_scores(method)
