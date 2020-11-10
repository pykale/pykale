"""
Digits dataset loading for EPIC-Kitchen, ADL, GTEA, KITCHEN. The code is based on
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/digits_dataset_access.py
"""

from enum import Enum
import kale.loaddata.video_data.video_transform as video_transform
from kale.loaddata.dataset_access import DatasetAccess
from kale.loaddata.video_data.epickitchen import EPIC
from copy import deepcopy


def get_videodata_config(cfg):
    config_params = {
        # "train_params": {
        #     "model": cfg.MODEL.METHOD,
        #     "init_lr": cfg.SOLVER.BASE_LR,
        #     "adapt_lr": cfg.SOLVER.BASE_LR,
        #     "nb_adapt_epochs": cfg.SOLVER.MAX_EPOCHS,
        #     "nb_init_epochs": cfg.SOLVER.MIN_EPOCHS,
        #     "tr_batch_size": cfg.SOLVER.TRAIN_BATCH_SIZE,
        #     "te_batch_size": cfg.SOLVER.TEST_BATCH_SIZE,
        #     "optimizer": {
        #         "type": cfg.SOLVER.TYPE,
        #         "optim_params": {
        #             "momentum": cfg.SOLVER.MOMENTUM,
        #             "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
        #             "nesterov": cfg.SOLVER.NESTEROV
        #         }
        #     }
        # },
        "data_params": {
            "dataset_root": cfg.DATASET.ROOT,
            "dataset_src_name": cfg.DATASET.SOURCE,
            "dataset_src_trainlist": cfg.DATASET.SRC_TRAINLIST,
            "dataset_src_testlist": cfg.DATASET.SRC_TESTLIST,
            "dataset_tar_name": cfg.DATASET.TARGET,
            "dataset_tar_trainlist": cfg.DATASET.TAR_TRAINLIST,
            "dataset_tar_testlist": cfg.DATASET.TAR_TESTLIST,
            "dataset_mode": cfg.DATASET.MODE,
            "num_classes": cfg.DATASET.NUM_CLASSES,
            "window_len": cfg.DATASET.WINDOW_LEN
        }
    }
    return config_params


def generate_list(data_name, data_params_local, domain):
    if data_name == 'EPIC':
        dataset_path = data_params_local['dataset_root'] + 'EPIC/EPIC_KITCHENS_2018/'
        data_path = dataset_path + 'frames_rgb_flow/'
        train_listpath = dataset_path + 'annotations/' + data_params_local['dataset_{}_trainlist'.format(domain)]
        test_listpath = dataset_path + 'annotations/' + data_params_local['dataset_{}_testlist'.format(domain)]
    elif data_name in ['ADL', 'GTEA', 'KITCHEN']:
        dataset_path = data_params_local['dataset_root'] + data_name + '/'
        data_path = dataset_path + 'frames_rgb_flow/'
        train_listpath = dataset_path + 'annotations/' + data_params_local['dataset_trainlist']
        test_listpath = dataset_path + 'annotations/' + data_params_local['dataset_testlist']

    return data_path, train_listpath, test_listpath


class VideoDataset(Enum):
    EPIC = 'EPIC'
    ADL = 'ADL'
    GTEA = 'GTEA'
    KITCHEN = 'KITCHEN'

    # HMDB51 = 'HMDB51'

    # @staticmethod
    # def get_data_configs(cfg):
    #     data_name = cfg.dataset_group.upper()
    #     mode = cfg.dataset_mode
    #     window_len = cfg.window_len
    #     num_classes = cfg.num_classes
    #     if data_name == 'EPIC':
    #         dataset_path = cfg.dataset_root + 'EPIC/EPIC_KITCHENS_2018/'
    #         data_path = dataset_path + 'frames_rgb_flow/'
    #         train_listpath = dataset_path + 'annotations/' + cfg.dataset_trainlist
    #         test_listpath = dataset_path + 'annotations/' + cfg.dataset_testlist
    #     elif data_name in ['ADL', 'GTEA', 'KITCHEN']:
    #         dataset_path = cfg.dataset_root + data_name + '/'
    #         data_path = dataset_path + 'frames_rgb_flow/'
    #         train_listpath = dataset_path + 'annotations/' + cfg.dataset_trainlist
    #         test_listpath = dataset_path + 'annotations/' + cfg.dataset_testlist
    #
    #     return data_path, train_listpath, test_listpath, mode, window_len, num_classes

    # Originally get_access
    # @staticmethod
    # def get_data(data: 'VideoDataset', params):
    #     # data: 'VideoDataset', data_path, train_list,
    #     #      test_list, mode, window_len, n_classes):
    #     """Gets data loaders for video datasets
    #
    #     Args:
    #         data (VideoDataset): dataset name
    #
    #     Examples::
    #         >>> data, num_channel = get_data(dataname, data_path, train_listpath, test_listpath, mode, window_len, n_classes)
    #     """
    #     config_params = get_videodata_config(params)
    #     data_params = config_params['data_params']
    #     data_params_local = deepcopy(data_params)
    #     data_name = data_params_local['dataset_group'].upper()
    #     mode = data_params_local['dataset_mode']
    #     window_len = data_params_local['window_len']
    #     num_classes = data_params_local['num_classes']
    #     if data_name == 'EPIC':
    #         dataset_path = data_params_local['dataset_root'] + 'EPIC/EPIC_KITCHENS_2018/'
    #         data_path = dataset_path + 'frames_rgb_flow/'
    #         train_listpath = dataset_path + 'annotations/' + data_params_local['dataset_trainlist']
    #         test_listpath = dataset_path + 'annotations/' + data_params_local['dataset_testlist']
    #     elif data_name in ['ADL', 'GTEA', 'KITCHEN']:
    #         dataset_path = data_params_local['dataset_root'] + data_name + '/'
    #         data_path = dataset_path + 'frames_rgb_flow/'
    #         train_listpath = dataset_path + 'annotations/' + data_params_local['dataset_trainlist']
    #         test_listpath = dataset_path + 'annotations/' + data_params_local['dataset_testlist']
    #
    #     channel_numbers = {
    #         VideoDataset.EPIC: 3,
    #     }
    #
    #     transform_names = {
    #         (VideoDataset.EPIC, 3): 'epic',
    #     }
    #
    #     factories = {
    #         VideoDataset.EPIC: EPICDatasetAccess,
    #     }
    #
    #     # handle color/nb channels
    #     num_channels = channel_numbers[data]
    #     data_tf = transform_names[(data, num_channels)]
    #
    #     return factories[data](data_path, train_listpath, test_listpath, mode, window_len, num_classes, data_tf)

    @staticmethod
    def get_source_target(source: "VideoDataset", target: "VideoDataset", params):
        config_params = get_videodata_config(params)
        data_params = config_params['data_params']
        data_params_local = deepcopy(data_params)
        data_src_name = data_params_local['dataset_src_name'].upper()
        src_data_path, src_tr_listpath, src_te_listpath = generate_list(data_src_name, data_params_local, domain='src')
        data_tar_name = data_params_local['dataset_tar_name'].upper()
        tar_data_path, tar_tr_listpath, tar_te_listpath = generate_list(data_tar_name, data_params_local, domain='tar')
        mode = data_params_local['dataset_mode']
        num_classes = data_params_local['num_classes']
        window_len = data_params_local['window_len']

        channel_numbers = {
            VideoDataset.EPIC: 3,
        }

        transform_names = {
            (VideoDataset.EPIC, 3): 'epic',
        }

        factories = {
            VideoDataset.EPIC: EPICDatasetAccess,
        }

        # handle color/nb channels
        num_channels = max(channel_numbers[source], channel_numbers[target])
        source_tf = transform_names[(source, num_channels)]
        target_tf = transform_names[(target, num_channels)]

        return (
            factories[source](src_data_path, src_tr_listpath, src_te_listpath, mode, window_len, num_classes,
                              source_tf),
            factories[target](tar_data_path, tar_tr_listpath, tar_te_listpath, mode, window_len, num_classes,
                              target_tf),
            num_channels
        )


class VideoDatasetAccess(DatasetAccess):
    """Common API for video dataset access

    Args:
        data_path (string): root directory of dataset
        transform_kind (string): types of video transforms
    """

    def __init__(self, data_path, train_list, test_list,
                 mode, window_len, n_classes, transform_kind):
        super().__init__(n_classes)
        self._data_path = data_path
        self._train_list = train_list
        self._test_list = test_list
        self._mode = mode
        self._window_len = window_len
        self._transform = video_transform.get_transform(transform_kind)


class EPICDatasetAccess(VideoDatasetAccess):
    """
    EPIC data loader
    """

    def get_train(self):
        return EPIC(
            data_path=self._data_path,
            list_path=self._train_list,
            mode=self._mode,
            window_len=self._window_len,
            dataset_split='train',
            transforms=self._transform['train']
        )

    def get_test(self):
        return EPIC(
            data_path=self._data_path,
            list_path=self._test_list,
            mode=self._mode,
            window_len=self._window_len,
            dataset_split='test',
            transforms=self._transform['test']
        )


class GTEADatasetAccess(VideoDatasetAccess):
    def get_train(self):
        return 0

    def get_test(self):
        return 0
