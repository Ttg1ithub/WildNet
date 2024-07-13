"""
Network Initializations
"""

import logging
import importlib
import torch
import datasets



def get_net(args, criterion, criterion_aux=None, cont_proj_head=0, wild_cont_dict_size=0):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(args=args, num_classes=datasets.num_classes,
                    criterion=criterion, criterion_aux=criterion_aux, cont_proj_head=cont_proj_head, wild_cont_dict_size=wild_cont_dict_size)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.3f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def warp_network_in_dataparallel(net, gpuid):
    """
    Wrap the network in Dataparallel
    """
    # torch.cuda.set_device(gpuid)
    # net.cuda(gpuid)    
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid], find_unused_parameters=True)
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid])#, find_unused_parameters=True)
    return net


def get_model(args, num_classes, criterion, criterion_aux=None, cont_proj_head=0, wild_cont_dict_size=0):
    """
    根据参数获取网络模型函数指针

    Args:
    - args: 命令行参数对象
    - num_classes: 分类的类别数目
    - criterion: 主要损失函数
    - criterion_aux: 辅助损失函数，默认为None
    - cont_proj_head: 连续投影头部设置，默认为0
    - wild_cont_dict_size: 野外连续字典大小，默认为0

    Returns:
    - net: 实例化后的网络模型
    """
    network = args.arch  # 获取网络架构描述
    module = network[:network.rfind('.')]  # 提取模块路径
    model = network[network.rfind('.') + 1:]  # 提取模型名称
    mod = importlib.import_module(module)  # 动态导入模块
    net_func = getattr(mod, model)  # 获取模块中的模型函数指针
    # 调用模型函数，实例化网络模型
    net = net_func(args=args, num_classes=num_classes, criterion=criterion,
                   criterion_aux=criterion_aux, cont_proj_head=cont_proj_head,
                   wild_cont_dict_size=wild_cont_dict_size)
    return net
