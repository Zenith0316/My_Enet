# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-06 22:50:12
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-03 21:50:38
# @Email:  cshzxie@gmail.com


class AverageMeter(object):
    """累计一个或多个指标的当前值、总和、计数和平均值。"""
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        """清空累计状态。"""
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        """加入一个标量或与 items 等长的指标列表。"""
        if type(values).__name__ == 'list':
            for idx, v in enumerate(values):
                self._val[idx] = v
                self._sum[idx] += v
                self._count[idx] += 1
        else:
            self._val[0] = values
            self._sum[0] += values
            self._count[0] += 1

    def val(self, idx=None):
        """返回最近一次写入的值。"""
        if idx is None:
            return self._val[0] if self.items is None else [self._val[i] for i in range(self.n_items)]
        else:
            return self._val[idx]

    def count(self, idx=None):
        """返回累计次数。"""
        if idx is None:
            return self._count[0] if self.items is None else [self._count[i] for i in range(self.n_items)]
        else:
            return self._count[idx]

    def avg(self, idx=None):
        """返回累计平均值。调用前需确保对应指标至少更新过一次。"""
        if idx is None:
            return self._sum[0] / self._count[0] if self.items is None else [
                self._sum[i] / self._count[i] for i in range(self.n_items)
            ]
        else:
            return self._sum[idx] / self._count[idx]
