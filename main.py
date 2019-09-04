# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time

import torch
import torch.nn.functional as F
from torch.autograd import grad

from data import CIFAR10, IMGNET12, MNIST
from vulnerability import compute_vulnerability
from utils import argument_parser, create_net, initialize_params
from penalties import addPenalty, pgd


# NB: Logger cannot be pushed to utils.py, because of eval(name)
class Logger(object):
    def __init__(self):
        self.logs = dict()

    def log(self, step, delta_time, *args):
        for name in args:
            if type(name) != str:
                raise Exception(
                    "Logger takes strings as inputs. "
                    "But got %s" % type(name))
            if name not in self.logs:
                self.logs[name] = []
            self.logs[name].append([eval(name), step, delta_time])

    def get_logs(self):
        return self.logs

    def set_logs(self, logs):
        self.logs = logs  # logs : dict
        return


def grad_norms(loss, inputs, train=False):
    bs = inputs.size(0)
    g = grad(loss, inputs, retain_graph=train)[0] * bs
    g = g.view(bs, -1)
    norm1, norm2 = g.norm(1, 1).mean(), g.norm(2, 1).mean()
    return norm1.item(), norm2.item()


def do_epoch(epoch, net, optimizer, loader, mode, args):
    if mode not in {'train', 'eval', 'test', 'init'}:
        # 'init'  -> for initialization of batchnorms
        # 'train' -> training (but no logging of vul & dam)
        # 'eval'  -> compute acc & gnorms but not vul & dam on validation
        # 'test'  -> compute all logged values on test set
        raise Exception('Argument mode must be train, eval or init')

    net.eval() if mode in {'eval', 'test'} else net.train()
    device = next(net.parameters()).device
    cum_loss = cum_pen = cum_norm1 = cum_norm2 = total = correct = 0.
    advVul = advCorrect = cum_dam = 0.
    predictedAdv = None

    for i, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad = True
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        norm1, norm2 = grad_norms(loss, inputs, mode == 'train')

        if mode == 'train':
            if args.lam > 0.:
                penalty = addPenalty(net, inputs, outputs, targets, loss, args)
                loss += penalty
                cum_pen += penalty.item()
                cum_loss += loss.item()
            loss.backward()
            optimizer.step()

        elif mode == 'test':  # compute adv vul & damage using custom PGD
            eps = .004
            advDam, advOutputs = pgd(
                net, inputs, targets, loss, lam=eps, steps=10,
                step_size=eps / (.75 * 10), random_start=False, train=False)

        # Compute logging info
        cum_norm1 += norm1
        cum_norm2 += norm2
        cum_loss += loss.item()
        total += targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).float().cpu().sum().item()
        if mode == 'test':
            cum_dam += advDam.item() / eps
            _, predictedAdv = torch.max(advOutputs.data, 1)
            advVul += predicted.size(0) - (
                predictedAdv.eq(predicted.data).float().cpu().sum().item())
            advCorrect += predictedAdv.eq(
                targets.data).float().cpu().sum().item()

        results = {
            'acc': 100 * correct / total,   # accuracy
            'loss': cum_loss / (i + 1),       # loss
            'pen': cum_pen / (i + 1),         # penalty
            'norm1': cum_norm1 / (i + 1),     # avg l1-gradient norm
            'norm2': cum_norm2 / (i + 1),     # avg l2-gradient norm
            'av': 100 * advVul / total,     # adversarial vulnerability
            'da': cum_dam / (i + 1),          # adversarial damage
            'aa': 100 * advCorrect / total  # adversarial accuracy
        }

        if args.log_step is not None and i % args.log_step == 0:
            print("Epoch: %03d Batch: %04d Mode: %-5s Acc: %4.1f Loss: %4.2f "
                  "Pen: %5.3f gNorm1: %6.2f gNorm2: %6.3f Vul: %4.1f "
                  "Dam: %6.2f AdAcc %4.1f" % (
                      epoch, i, mode, *[
                          results[i] for i in ['acc', 'loss', 'pen', 'norm1',
                                               'norm2', 'av', 'da', 'aa']]))

    return results


if __name__ == '__main__':
    parser, args = argument_parser()
    logger = Logger()
    args.path = os.path.join('results', args.name)
    net = create_net(args)
    # print(net)

    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)  # requires Python >= 3.2

    if os.path.isfile(os.path.join(args.path, 'last.pt')):
        print('> Loading last saved state/network...')
        state = torch.load(os.path.join(args.path, 'last.pt'))
        net.load_state_dict(state['state_dict'])
        lr = state['lr']
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        optimizer.load_state_dict(state['optimizer'])
        best_va_acc = state['best_va_acc']
        start_ep = state['epoch'] + 1
        logger.set_logs(state['logs'])
    else:  # initialize new net
        print('> Initializing new network...')
        net.apply(lambda m: initialize_params(m, args.no_act, 'normal'))
        lr = args.lr
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        best_va_acc = -1.
        start_ep = -1
    print('> Done.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    torch.backends.cudnn.benchmark = True

    print('> Loading dataset...')
    if args.dataset == 'mnist':
        tr_loader, va_loader, te_loader = MNIST(
            root=args.datapath, bs=args.bs, valid_size=.1,
            size=args.img_size, normalize=(not args.raw_inputs))
    elif args.dataset == 'cifar':
        tr_loader, va_loader, te_loader = CIFAR10(
            root=args.datapath, bs=args.bs, valid_size=.1,
            size=args.img_size, normalize=(not args.raw_inputs))
    elif args.dataset == 'imgnet12':
        tr_loader, va_loader, te_loader = IMGNET12(
            root=args.datapath, bs=args.bs, valid_size=.1,
            size=args.img_size, normalize=(not args.raw_inputs))
    else:
        raise NotImplementedError
    print('> Done.')

    print('> Starting training.')
    time_start = time.time()
    epochs = 0 if args.no_training else args.epochs
    for epoch in range(start_ep, epochs):
        time_start = time.time()

        if epoch % 30 == 0 and epoch > 0:
            # reload best parameters on validation set
            net.load_state_dict(
                torch.load(os.path.join(
                    args.path, 'best.pt'))['state_dict'])
            # update learning rate
            lr *= .5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        mode = 'init' if epoch < 0 else 'train'
        tr_res = do_epoch(epoch, net, optimizer, tr_loader, mode, args)
        va_res = do_epoch(epoch, net, optimizer, va_loader, 'eval', args)
        te_res = do_epoch(epoch, net, optimizer, te_loader, 'test', args)

        time_per_epoch = time.time() - time_start

        print("epoch %3d lr %.1e te_norm1 %7.3f te_norm2 %6.4f tr_loss %6.3f "
              "tr_acc %5.2f te_acc %5.2f te_aa %5.2f te_av %5.2f te_da %6.3f "
              "va_acc %5.2f be_va_acc %5.2f time %d" % (
                  epoch, lr, te_res['norm1'], te_res['norm2'], tr_res['loss'],
                  tr_res['acc'], te_res['acc'], te_res['aa'], te_res['av'],
                  te_res['da'], va_res['acc'], best_va_acc,
                  time_per_epoch))

        # Log and save results
        logger.log(epoch, time_per_epoch, 'lr', 'tr_res', 'va_res', 'te_res')

        state = {
            'lr': lr,
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args,
            'logs': logger.get_logs(),
            'best_va_acc': best_va_acc
        }

        torch.save(state, os.path.join(args.path, 'last.pt'))

        if va_res['acc'] > best_va_acc:
            best_va_acc = va_res['acc']
            torch.save(state, os.path.join(args.path, 'best.pt'))

    print('> Finished Training')

    # Compute adversarial vulnerability with foolbox
    print('\n> Starting attacks.')
    attacks = {'l1'}
    # attacks = {'l1', 'l2', 'itl1', 'itl2', 'deepFool', 'pgd', 'boundary'}
    for attack in attacks:
        vulnerability = compute_vulnerability(
            args, attack, net, args.n_attacks)
        torch.save(vulnerability,
                   os.path.join(args.path, 'vulnerability_%s.pt' % attack))
