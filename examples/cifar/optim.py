# Created by Haiping Lu from modifying https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/optim.py
# Under the MIT License
# To remove
import torch.optim as optim
from config import C


def construct_optim(net):
    # SReLU parameters.
    srelu_params = []
    # channel-shared parameters.
    shared_params = []
    # Non-batchnorm parameters.
    other_params = []
    for name, p in net.named_parameters():
        if 'srelu' in name:
            srelu_params.append(p)
        elif 'shared' in name:
            shared_params.append(p)
        else:
            other_params.append(p)

    optim_params = [
        {
            'params': srelu_params,
            'weight_decay': 0.0,
        },
        {
            'params': other_params,
            'weight_decay': C.SOLVER.WEIGHT_DECAY,
        },
        {
            'params': shared_params,
            'lr': C.SOLVER.BASE_LR / 10,
            'scaling': 0.1,
        }
    ]
    # Check all parameters will be passed into optimizer.
    assert len(list(net.parameters())) == len(other_params) + len(srelu_params) + len(shared_params), \
        f'parameter size does not match: ' \
        f'{len(other_params)} + {len(srelu_params)} + {len(shared_params)} != ' \
        f'{len(list(net.parameters()))}'

    return optim.SGD(
        optim_params,
        lr=C.SOLVER.BASE_LR,
        momentum=C.SOLVER.MOMENTUM,
        weight_decay=C.SOLVER.WEIGHT_DECAY,
        dampening=C.SOLVER.DAMPENING,
        nesterov=C.SOLVER.NESTEROV
    )
