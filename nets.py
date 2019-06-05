# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from functools import reduce

import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FlexibleAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return F.avg_pool2d(inputs, kernel_size=inputs.size(2))


class WeightPool(nn.Module):
    def __init__(self, in_planes, kernel_size):
        super(WeightPool, self).__init__()
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size,
                              stride=kernel_size, groups=in_planes, bias=False)
        self.conv.unit_gain = True

    def forward(self, x):
        return self.conv(x)


class WeightPoolOut(nn.Module):
    def __init__(self, in_planes, plane_size, categories, unit_gain=False):
        super(WeightPoolOut, self).__init__()
        self.in_planes = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=plane_size,
                              groups=in_planes, bias=False)
        self.linear = nn.Linear(in_planes, categories)
        self.linear.unit_gain = unit_gain

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, self.in_planes)
        return self.linear(out)


class MaxPoolOut(nn.Module):
    def __init__(self, in_planes, plane_size, categories, unit_gain=False):
        super(MaxPoolOut, self).__init__()
        self.in_planes = in_planes
        self.maxpool = nn.MaxPool2d(kernel_size=plane_size)
        self.linear = nn.Linear(in_planes, categories)
        self.linear.unit_gain = unit_gain

    def forward(self, x):
        out = self.maxpool(x)
        out = out.view(-1, self.in_planes)
        return self.linear(out)


class AvgPoolOut(nn.Module):
    def __init__(self, in_planes, plane_size, categories, unit_gain=False):
        super(AvgPoolOut, self).__init__()
        self.in_planes = in_planes
        self.avgpool = nn.AvgPool2d(kernel_size=plane_size)
        self.linear = nn.Linear(in_planes, categories)
        self.linear.unit_gain = unit_gain

    def forward(self, x):
        out = self.avgpool(x)
        out = out.view(-1, self.in_planes)
        return self.linear(out)


class FCout(nn.Module):
    def __init__(self, in_planes, plane_size, categories, unit_gain=False):
        super(FCout, self).__init__()
        if type(plane_size) == tuple and len(plane_size) == 2:
            plane_size = reduce(lambda x, y: x * y, plane_size)
        else:
            plane_size = plane_size ** 2
        print('Plane size = ', plane_size)

        self.in_planes = in_planes
        self.plane_size = plane_size
        self.linear = nn.Linear(in_planes * plane_size, categories)
        self.linear.unit_gain = unit_gain

    def forward(self, x):
        out = x.view(-1, self.in_planes * self.plane_size)
        return self.linear(out)


class ConvLayer(nn.Module):
    def __init__(self, in_planes, planes, pooltype=None, no_BN=False,
                 no_act=False, dilation=1):
        super(ConvLayer, self).__init__()
        self.pad = nn.ReflectionPad2d(dilation)
        if pooltype is None:  # Usual conv
            self.conv = nn.Conv2d(in_planes, planes, 3, padding=0,
                                  stride=1, dilation=dilation)
        elif pooltype == 'avgpool':  # Average Pool
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, planes, 3, dilation=dilation),
                nn.AvgPool2d(2))
        elif pooltype == 'subsamp':  # Strided Conv
            self.conv = nn.Conv2d(
                in_planes, planes, 3, stride=2, dilation=dilation)
        elif pooltype == 'maxpool':  # Max Pool
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, planes, 3, dilation=dilation),
                nn.MaxPool2d(2))
        elif pooltype == 'weightpool':
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, planes, 3, dilation=dilation),
                WeightPool(planes, 2))
        else:
            raise NotImplementedError
        if no_act:
            self.act = lambda x: x
        else:
            self.act = nn.ReLU()
        if no_BN:
            self.bn = lambda x: x  # Identity()
        else:
            self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.act(self.bn(self.conv(self.pad(x))))
        return out


class ConvNet(nn.Module):
    def __init__(
            self, categories=10, n_layers=3, in_size=32, poolings=None,
            pooltype='avgpool', no_BN=False, no_act=False, dilations=1,
            normalize_inputs=False, last_layers='maxpool', in_planes=3):

        # last_layers in {'maxpool', 'fc', 'weightpool'}

        super(ConvNet, self).__init__()
        poolings = [] if poolings is None else poolings
        if type(dilations) != list:
            dilations = [dilations] * n_layers
        self.in_planes = in_planes

        if normalize_inputs or no_BN:
            self.bn = (lambda x: x)
        else:
            self.bn = nn.BatchNorm2d(self.in_planes)

        self.layers = self._make_layers(
            ConvLayer, 64, n_layers, poolings, pooltype,
            no_BN, no_act, dilations)

        # compute input-size to last layers from input-size of the net
        # self.in_planes is changed by _make_layers to the nbr of out-planes
        out_size = int(in_size / (2 ** (len(poolings))))

        self.last_layers = self._make_last_layers(
            out_size, categories, last_layers)

    def _make_layers(self, block, planes, num_blocks, poolings,
                     pooltype, no_BN, no_act, dilations):
        # pooltypes = [0] + [0] * (num_blocks - 1)
        pooltypes = [None] * num_blocks
        for pool in poolings:
            pooltypes[pool] = pooltype
        layers = []
        for pool, dilation in zip(pooltypes, dilations):
            layers.append(block(self.in_planes, planes, pool, no_BN, no_act,
                                dilation))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _make_last_layers(self, in_size, categories, last_layers):
        if last_layers == 'maxpool':
            last_layers = MaxPoolOut(
                self.in_planes, in_size, categories, unit_gain=True)
        elif last_layers == 'avgpool':
            last_layers = AvgPoolOut(
                self.in_planes, in_size, categories, unit_gain=True)
        elif last_layers == 'weightpool':
            last_layers = WeightPoolOut(
                self.in_planes, in_size, categories, unit_gain=True)
        elif last_layers == 'fc':
            last_layers = FCout(
                self.in_planes, in_size, categories, unit_gain=True)
        else:
            raise NotImplementedError(
                'Argument last_layers must be maxpool, fc, weightpool. '
                'But got: %s' % last_layers)

        return last_layers

    def forward(self, x):
        out = self.layers(self.bn(x))
        out = self.last_layers(out)
        return out
