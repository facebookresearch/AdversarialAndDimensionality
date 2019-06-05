# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch
import torch.nn.functional as F
from torch.autograd import grad


def gPenalty(inputs, loss, lam, q):
    # Gradient penalty
    bs, c, h, w = inputs.size()
    d_in = c * h * w
    g = grad(loss, inputs, create_graph=True)[0] * bs
    g = g.view(bs, -1)
    qnorms = g.norm(q, 1).mean()
    lam = lam * math.pow(d_in, 1. - 1. / q)
    return lam * qnorms.mean() / 2.


def advAugment(net, inputs, targets, loss, lam, q):
    # Single-step adversarial augmentation (e.g. FGSM)
    bs, c, h, w = inputs.size()
    d_in = c * h * w
    g = grad(loss, inputs, retain_graph=True)[0] * bs
    g = g.view(bs, -1).detach()
    if q == 1:
        lam = lam
        dx = lam * g.sign()
    else:
        p = 1. / (1. - 1. / q)
        lam = lam * math.pow(d_in, 1. - 1. / q)
        dx = g.sign() * g.abs().pow(q - 1)  # sign when q uneven
        pnorms = dx.norm(p, 1, keepdim=True)
        dx = lam * dx / pnorms
    dx = dx.view_as(inputs)
    advInputs = (inputs + dx).detach()
    advOutputs = net(advInputs)
    advLoss = F.cross_entropy(advOutputs, targets)
    return (advLoss - loss) / 2.


def pgd(net, inputs, targets, loss, lam, steps, step_size,
        random_start=True, train=True):
    # Projected gradient descent (i.e. iterative FGSM) with random starts
    bs, c, h, w = inputs.size()
    if random_start:
        if torch.cuda.is_available():
            noise = torch.cuda.FloatTensor(bs, c, h, w).uniform_(-lam, lam)
        else:
            noise = torch.FloatTensor(bs, c, h, w).uniform_(-lam, lam)
    else:
        if torch.cuda.is_available():
            noise = torch.cuda.FloatTensor(bs, c, h, w).fill_(0)
        else:
            noise = torch.FloatTensor(bs, c, h, w).fill_(0)
    advInputs = (inputs + noise).detach()
    advInputs.requires_grad = True
    advOutputs = net(advInputs)
    advLoss = F.cross_entropy(advOutputs, targets)
    for i in range(steps):
        retain_graph = ((i + 1 == steps) and train)
        g = grad(advLoss, advInputs, retain_graph=retain_graph)[0] * bs
        g = g.view(bs, -1).detach()
        dx = step_size * g.sign()
        dx = dx.view_as(advInputs)
        advInputs = advInputs + dx
        advInputs = inputs + torch.clamp(advInputs - inputs, -lam, lam)
        advInputs = advInputs.detach()
        advInputs.requires_grad = True
        advOutputs = net(advInputs)
        advLoss = F.cross_entropy(advOutputs, targets)
    return advLoss - loss, advOutputs


def crossLip(inputs, outputs, lam):
    gk = []
    n, K, cLpen = outputs.size(0), outputs.size(1), 0.
    for k in range(K):
        gk.append(grad(outputs[:, k].sum(), inputs, create_graph=True)[0])
    for l in range(K):
        for m in range(l + 1, K):
            cLpen += (gk[l] - gk[m]) ** 2
    cLpen = 2. / n / K ** 2 * cLpen.sum()
    return lam * cLpen


def addPenalty(net, inputs, outputs, targets, loss, args):
    if args.penalty == 'grad':
        penalty = gPenalty(inputs, loss, args.lam, args.q)
    elif args.penalty == 'adv':
        penalty = advAugment(net, inputs, targets, loss, args.lam, args.q)
    elif args.penalty == 'pgd':
        penalty, _ = pgd(  # uses linf attacks
            net, inputs, targets, loss, args.lam,
            args.steps, step_size=args.lam / (.75 * args.steps))
    elif args.penalty == 'crossLip':
        penalty = crossLip(inputs, outputs, args.lam)
    else:
        raise NotImplementedError("Unknown penalty %r" % args.penalty)
    return penalty
