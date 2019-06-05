# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as transforms
import torchvision.datasets as datasets


def IMGNET12(root='~/datasets/imgnet12/', bs=32, bs_test=None, num_workers=32,
             valid_size=.1, size=256, crop=False, normalize=False):

    # Datafolder '~/datasets/imgnet12/' should contain folders train/ and val/,
    # each of which whould contain 12 subfolders (1 per class) with .jpg files

    root = os.path.expanduser(root)

    # original means = [.485, .456, .406]
    # original stds = [0.229, 0.224, 0.225]

    means = [.453, .443, .403]
    stds = {
        256: [.232, .226, .225],
        128: [.225, .218, .218],
        64: [.218, .211, .211],
        32: [.206, .200, .200]
    }

    if normalize:
        normalize = transforms.Normalize(mean=means,
                                         std=stds[size])
    else:
        normalize = transforms.Normalize((0., 0., 0),
                                         (1., 1., 1.))

    if bs_test is None:
        bs_test = bs

    if crop:
        tr_downsamplingOp = transforms.RandomCrop(size)
        te_downsamplingOp = transforms.CenterCrop(size)
    else:
        tr_downsamplingOp = transforms.Resize(size)
        te_downsamplingOp = transforms.Resize(size)

    preprocess = [transforms.Resize(256), transforms.CenterCrop(256)]

    tr_transforms = transforms.Compose([
        *preprocess,
        tr_downsamplingOp,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize, ])

    te_transforms = transforms.Compose([
        *preprocess,
        te_downsamplingOp,
        transforms.ToTensor(),
        normalize, ])

    tr_dataset = datasets.ImageFolder(root + '/train', transform=tr_transforms)
    te_dataset = datasets.ImageFolder(root + '/val', transform=te_transforms)

    # Split training in train and valid set
    num_train = len(tr_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    tr_idx, va_idx = indices[split:], indices[:split]

    tr_sampler = SubsetRandomSampler(tr_idx)
    va_sampler = SubsetRandomSampler(va_idx)

    tr_loader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=bs,
        num_workers=num_workers, pin_memory=True, sampler=tr_sampler)

    va_loader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=bs_test,
        num_workers=num_workers, pin_memory=True, sampler=va_sampler)

    te_loader = torch.utils.data.DataLoader(
        te_dataset, batch_size=bs_test, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    if valid_size > 0.:
        return tr_loader, va_loader, te_loader
    else:
        return tr_loader, te_loader


def CIFAR10(root='~/datasets/cifar10/', bs=128, bs_test=None,
            augment_training=True, valid_size=0., size=32, num_workers=1,
            normalize=False):
    root = os.path.expanduser(root)

    if bs_test is None:
        bs_test = bs

    if normalize:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
    else:
        normalize = transforms.Normalize((0., 0., 0),
                                         (1., 1., 1.))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size, Image.NEAREST),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size, Image.NEAREST),
        transforms.ToTensor(),
        normalize
    ])

    transform_valid = transform_test

    if augment_training is False:
        transform_train = transform_test

    dataset_tr = datasets.CIFAR10(root=root,
                                  train=True,
                                  transform=transform_train)

    dataset_va = datasets.CIFAR10(root=root,
                                  train=True,
                                  transform=transform_valid)

    dataset_te = datasets.CIFAR10(root=root,
                                  train=False,
                                  transform=transform_test)

    # Split training in train and valid set
    num_train = len(dataset_tr)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    loader_tr = torch.utils.data.DataLoader(dataset_tr,
                                            batch_size=bs,
                                            sampler=train_sampler,
                                            num_workers=num_workers)

    loader_va = torch.utils.data.DataLoader(dataset_va,
                                            batch_size=bs,
                                            sampler=valid_sampler,
                                            num_workers=num_workers)

    # add pin_memory
    loader_te = torch.utils.data.DataLoader(dataset_te,
                                            batch_size=bs_test,
                                            shuffle=False,
                                            num_workers=num_workers)
    if valid_size > 0:
        return loader_tr, loader_va, loader_te
    else:
        return loader_tr, loader_te


def MNIST(root='~/datasets/mnist/', bs=128, bs_test=None,
          augment_training=True, valid_size=0., size=32, num_workers=1,
          normalize=False):
    root = os.path.expanduser(root)

    if bs_test is None:
        bs_test = bs

    if normalize:
        normalize = transforms.Normalize((0.1307,), (0.3081,))
    else:
        normalize = transforms.Normalize((0.,), (1.,))

    transform = transforms.Compose([
        transforms.Resize(32, Image.BILINEAR),
        transforms.Resize(size, Image.NEAREST),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        normalize
    ])

    dataset_tr = datasets.MNIST(root=root,
                                train=True,
                                transform=transform)

    dataset_va = datasets.MNIST(root=root,
                                train=True,
                                transform=transform)

    dataset_te = datasets.MNIST(root=root,
                                train=False,
                                transform=transform)

    # Split training in train and valid set
    num_train = len(dataset_tr)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    loader_tr = torch.utils.data.DataLoader(dataset_tr,
                                            batch_size=bs,
                                            sampler=train_sampler,
                                            num_workers=num_workers)

    loader_va = torch.utils.data.DataLoader(dataset_va,
                                            batch_size=bs,
                                            sampler=valid_sampler,
                                            num_workers=num_workers)

    # add pin_memory
    loader_te = torch.utils.data.DataLoader(dataset_te,
                                            batch_size=bs_test,
                                            shuffle=False,
                                            num_workers=num_workers)
    if valid_size > 0:
        return loader_tr, loader_va, loader_te
    else:
        return loader_tr, loader_te
