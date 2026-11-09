"""
Weight initialization utilities for Pathomic Fusion.
"""

import math
import torch
import torch.nn as nn
from torch.nn import init


def init_weights(net, init_type='orthogonal', init_gain=0.02):
    """
    Initialize network weights.
    
    Args:
        net: Network to be initialized
        init_type: Initialization method (normal | xavier | kaiming | orthogonal)
        init_gain: Scaling factor
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
            classname.find('Conv') != -1 or classname.find('Linear') != -1
        ):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'max':
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.normal_(0, stdv)
            else:
                raise NotImplementedError(
                    f'Initialization method [{init_type}] is not implemented'
                )
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f'Initializing network with {init_type}')
    net.apply(init_func)


def init_max_weights(module):
    """
    Self-Normalizing Network (SNN) weight initialization.
    Uses normal distribution with stdv = 1/sqrt(fan_in).
    """
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Initialize a network:
    1. Register CPU/GPU device (with multi-GPU support)
    2. Initialize the network weights
    
    Args:
        net: The network to be initialized
        init_type: Initialization method
        init_gain: Scaling factor
        gpu_ids: GPU device IDs
        
    Returns:
        Initialized network
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available(), 'CUDA is not available'
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)

    if init_type not in ('max', 'none'):
        print(f"Init Type: {init_type}")
        init_weights(net, init_type, init_gain=init_gain)
    elif init_type == 'none':
        print("Init Type: Not initializing networks.")
    elif init_type == 'max':
        print("Init Type: Self-Normalizing Weights")

    return net
