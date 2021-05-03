import logging
import os

# from kale.utils import logger
from kale.utils.csv_logger import setup_logger


def test_csv_logger(download_path):
    train_params = {"param1": 1, "param2": 2}
    method = "DANN"
    seed = 32  # To define in conftest later?
    testing_logger, results, _, test_csv_file = setup_logger(train_params, download_path, method, seed)
    assert isinstance(testing_logger, logging.Logger)

    # metrics = {"metric1": [1, 2], "metric2": [3, 4]}
    # df_metrics = pd.DataFrame(data=metrics)
    # results.update(
    #         is_validation=False, method_name=method, seed=seed, metric_values=df_metrics,
    #     )
    # results.print_score(method)
    results.append_to_txt(os.path.join(download_path, "temp.txt"), train_params, 3)
    results.append_to_markdown(os.path.join(download_path, "temp.md"), train_params, 3)
    results.to_csv(test_csv_file)
    assert os.path.isfile(test_csv_file)
    assert os.path.isfile(os.path.join(download_path, "temp.txt"))
    assert os.path.isfile(os.path.join(download_path, "temp.md"))

    # Teardown log file
    os.remove(test_csv_file)
    os.remove(os.path.join(download_path, "parameters.json"))
    os.remove(os.path.join(download_path, "temp.txt"))
    os.remove(os.path.join(download_path, "temp.md"))
