# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

""" Some utilities """
import os
import math
import warnings

import configargparse
import torch

from nets import ConvNet


def argument_parser():
    parser = configargparse.ArgParser(
        description='First-order vulnerability and input dimension')

    parser.add(
        '--config', required=True, is_config_file=True,
        help='configuration file path')
    parser.add_argument(
        '--name', type=str,
        help='Experiment name. Results will be saved/loaded from directory '
             './results/name (which will be created if needed).')
    parser.add_argument(
        '--datapath', type=str, default=None,
        help="Data location. Default: '~/datasets/' + `dataset`")
    parser.add_argument(
        '--dataset', type=str, default='cifar',
        help='mnist, cifar, imgnet12 (default: cifar)')
    parser.add_argument(
        '--img_size', type=int, default=None,
        help='only for imgnet. Resize img to 32, 64, 128 or 256.')
    parser.add_argument(
        '--n_layers', type=int, default=5,
        help='number of hidden layers')
    parser.add_argument(
        '--bs', type=int, default=128,
        help='batch size')
    parser.add_argument(
        '--epochs', type=int, default=200,
        help='number of training epochs')
    parser.add_argument(
        '--no_BN', action='store_true',
        help='Do not use batch norms (except before the very 1st layer)')
    parser.add_argument(
        '--no_act', action='store_true',
        help='No activation functions (f.ex. no ReLUs)')
    parser.add_argument(
        '--raw_inputs', action='store_true',
        help='Do not normalize inputs (hence no bn as first network layer)')
    parser.add_argument(
        '--log_step', type=int, default=None,
        help='print training info every log_step batches (default: None)')

    # training
    parser.add_argument(
        '--lr', type=float, default=.01,
        help='Initial learning rate')
    parser.add_argument(
        '--no_training', action='store_true',
        help='Do not train the network')
    parser.add_argument(
        '--crop', action='store_true',
        help='Use cropping instead of resizing image.')

    # Penalties/Regularizers
    penalties = ['grad', 'adv', 'pgd', 'crossLip']
    parser.add_argument(
        '--lam', type=float, default=0.,
        help='global regularization weight')
    parser.add_argument(
        '--penalty', type=str, choices=penalties, default=None,
        help='penalty type:' + ' | '.join(penalties))
    parser.add_argument(
        '--q', type=int, default=None,
        help="defense-norm q; dual of attack-norm p. "
             "For FGSM, use penalty='adv' and 'q=1'")
    parser.add_argument(
        '--steps', type=int, default=None,
        help='number of optimization steps per attack when using PGD')

    # Vulnerability.py specific
    parser.add_argument(
        '--n_attacks', type=int, default=-1,
        help='number of attack iterations; -1 for whole dataset')
    parser.add_argument(
        '--log_vul', action='store_true',
        help='Print detailed logs of vulnerability computation')

    # ConvNet specific
    pooltypes = ['avgpool', 'maxpool', 'weightpool', 'subsamp']
    last_layers = ['maxpool', 'avgpool', 'fc', 'weightpool']
    parser.add_argument(
        '--poolings', nargs='*', type=int, default=[],
        help='Where to do poolings. Should be a list of '
             'integers smaller than n_layers. Defaults to None. (ConvNet)')
    parser.add_argument(
        '--pooltype', type=str,
        choices=pooltypes, default='subsamp',
        help='penalty type:' + ' | '.join(penalties) + 'default: subsamp')
    parser.add_argument(
        '--dilations', nargs='*', type=int, default=None,
        help='Dilations to use for each layer. List of n_layers int. '
             'Defaults to 1 for all layers. (ConvNet)')
    parser.add_argument(
        '--last_layers', type=str, choices=last_layers,
        default='avgpool', help='penalty type:' + ' | '.join(last_layers))

    args = parser.parse_args()

    if args.datapath is None:
        args.datapath = os.path.join('~/datasets/', args.dataset)
    args.datapath = os.path.expanduser(args.datapath)

    # DATASET SPECIFIC SETTINGS
    if args.dataset == 'mnist':
        if args.img_size is None:
            args.img_size = 32
        elif args.img_size not in {32, 64, 128, 256, 512}:
            raise Exception(
                "img_size must be 32, 64, 128, 256. "
                "But provided %r" % args.img_size)
        args.categories = 10
        args.in_planes = 1

    elif args.dataset == 'cifar':
        if args.img_size is None:
            args.img_size = 32
        elif args.img_size not in {32, 64, 128, 256, 512}:
            raise Exception(
                "img_size must be 32, 64, 128, 256, or 512. "
                "But provided %r" % args.img_size)
        args.categories = 10
        args.in_planes = 3

    elif args.dataset == 'imgnet12':
        if args.img_size is None:
            args.img_size = 256
        elif args.img_size not in {32, 64, 128, 256}:
            raise Exception(
                "img_size must be 32, 64, 128, or 256. "
                "But provided %r" % args.img_size)
        if args.bs > 32:
            raise Exception(
                "With imgnet12, Batchsize bs should be <= 32. "
                "Otherwise, you'll probably run out of GPU memory")
        args.categories = 12
        args.in_planes = 3

    else:
        raise NotImplementedError("Dataset unknown")

    # NETWORK DOUBLE-CHECKS/WARNINGS
    if args.no_BN and args.raw_inputs:
        warnings.warn(
            "no_BN also removes the first BN layer before the net "
            "which serves as normalization of data when using raw_inputs. "
            "Thus data input data stays unnormalized between 0 and 1")

    if args.dilations is None:
        dilation = 1 if args.crop else int(args.img_size / 32)
        args.dilations = [dilation] * args.n_layers
    elif len(args.dilations) == 1:
        args.dilations = args.dilations * args.n_layers
    elif len(args.dilations) != args.n_layers:
        raise Exception(
            'Argument dilations must be single integer, or a list of '
            'integers of length n_layers')

    # PENALTY/REGULARIZATION WARNINGS
    if (args.lam, args.penalty, args.q) != (0., None, None):
        if args.lam == 0.:
            warnings.warn(
                "Arguments penalty and/or q are given, but lam = 0. "
                "Set lam > 0., otherwise not penalty is used")
        elif args.penalty is None:
            raise Exception("Argument lam > 0., but no penalty is defined.")
        elif (args.penalty in {'adv', 'grad'}) and (args.q is None):
            raise Exception(
                "If argument penalty is 'adv' or 'grad', q must be in "
                "[1, infty]")
    if (args.penalty == 'pgd') and (args.steps is None):
        raise Exception(
            "Arguments steps must be specified with "
            "penalty-option pgd")

    return parser, args


def create_net(args):
    net = ConvNet(
        args.categories, args.n_layers, args.img_size, args.poolings,
        args.pooltype, args.no_BN, args.no_act, args.dilations,
        normalize_inputs=(not args.raw_inputs),
        last_layers=args.last_layers, in_planes=args.in_planes)
    return net


def initialize_params(m, no_act=False, distribution='normal'):
    # gain = sqrt 2 for ReLU
    gain = 1. if no_act else math.sqrt(2)
    try:  # if last layer, then gain = 1.
        if m.unit_gain:  # test if module as attribute 'last'
            gain = 1.
    except AttributeError:
        pass

    if type(m) in {torch.nn.Conv2d, torch.nn.Linear}:
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.)
        out_ = m.weight.data.size(0)
        in_ = m.weight.data.view(out_, -1).size(1)
        sigma = gain / math.sqrt(in_)
        if distribution is 'uniform':
            xmax = math.sqrt(3) * sigma
            torch.nn.init.uniform_(m.weight, a=-xmax, b=xmax)
        elif distribution is 'normal':
            torch.nn.init.normal_(m.weight, std=sigma)
        else:
            raise NotImplementedError(
                "Argument distribution must be 'uniform' or 'normal'. "
                "Got: '%r'" % distribution)

    elif type(m) == torch.nn.BatchNorm2d:
        if m.affine:
            torch.nn.init.constant_(m.bias, 0.)
            torch.nn.init.constant_(m.weight, 1.)
        if m.track_running_stats:
            torch.nn.init.constant_(m.running_mean, 0.)
            torch.nn.init.constant_(m.running_var, 1.)
