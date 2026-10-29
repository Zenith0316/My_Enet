import argparse
import logging
import os
import numpy as np
import sys
import torch
from pprint import pprint
from config_vipc import cfg
from run.train import train_net
from run.test import test_net

# 指定 CUDA 设备的编号规则和本次程序可见的 GPU。
# cfg.CONST.DEVICE 通常是配置文件中的 GPU 编号，例如 "0" 或 "0,1"。
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE


def set_seed(seed):
    """为 NumPy 和 PyTorch 设置随机种子，使实验结果尽量可复现。"""
    # 固定 NumPy 在数据处理、采样等操作中使用的随机数序列。
    np.random.seed(seed)

    # 固定 PyTorch 在 CPU 上执行操作时使用的随机数序列。
    torch.manual_seed(seed)

    # 固定当前 GPU 以及所有 GPU 上的随机数序列。
    # manual_seed_all 对多 GPU 训练尤其重要。
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args_from_command_line():
    """解析命令行参数，返回包含运行模式的参数对象。"""
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')

    # action='store_true' 表示：命令行中出现该参数时值为 True，否则为 False。
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
    args = parser.parse_args()

    return args


def main():
    """程序主流程：展示环境与配置，并选择训练或测试流程。"""
    # 获取命令行参数。默认不传参数时进入训练模式。
    args = get_args_from_command_line()

    # 打印 PyTorch 是否检测到可用的 CUDA GPU，便于检查运行环境。
    print('cuda available ', torch.cuda.is_available())

    # 打印本次运行使用的完整配置，方便记录和排查实验参数。
    print('Use config:')
    pprint(cfg)

    # 既没有指定 --test，也没有指定 --inference 时，启动模型训练。
    if not args.test and not args.inference:
        train_net(cfg)
    else:
        # 测试或推理必须先加载已经训练好的模型权重。
        # 权重文件路径由配置项 cfg.CONST.WEIGHTS 提供。
        if cfg.CONST.WEIGHTS is None:
            raise Exception('Please specify the path to checkpoint in the configuration file!')

        # 当前项目将 --test 和 --inference 都交给 test_net 处理；
        # 两种模式的具体差异（如果有）由 test_net 或配置项决定。
        test_net(cfg)


if __name__ == '__main__':
    # 只有直接执行 main.py 时才运行以下初始化和主流程；
    # 当 main.py 被其他模块导入时，这部分代码不会执行。
    seed = 1
    set_seed(seed)

    # PIL 在读取图片时可能输出较多调试信息，这里只保留 WARNING 及以上级别。
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # 设置全局日志格式和日志级别。
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    main()
