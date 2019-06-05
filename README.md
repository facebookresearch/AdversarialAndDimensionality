First-order adversarial vulnerability of neural networks and input dimension: Code and Supplements
==================================================================================================

This repository contains code to accompany the ICML 2019 paper
[First-order adversarial vulnerability of neural networks and input dimension](http://proceedings.mlr.press/v97/simon-gabriel19a.html).

Packages
--------

Special packages used:
- pytorch (tested on 1.0.0)
- foolbox (tested on 2.0.0)
- configargparse

Quick-start
-----------
- Train a net with default configuration and compute its vulnerability:
```
    python main.py
```
- Train a net on CIFAR10 with PGD and compute its vulnerability, with
  simililar settings than Section 4.1:
```
    python main.py --config configs/config_cifar_pgd.txt
```
- Train a net on upsampled CIFAR10 (bCIFAR10) and compute its vulnerability,
  with similar settings than Section 4.2:

```
    python main.py --config configs/config_bcifar.txt
```
- Any parameter from the config file can be overwritten from the command line,
  as in:
```
    python main.py --config configs/config_bcifar.txt --img_size 64 --log_step 10
```
See files in _configs/_ for other typical experiment settings/arguments.

#### Remark:
The experiments in Section 4.2 actually do not use the vulnerabilities contained
in the _vulnerability_XX.pt_ files (computed using the `compute_vulnerability`
function from _vulnerability.py_). Instead, they use the test-set
vulnerabilities contained in state['logs'] of the file _last.pt_ (and computed
using our custom pgd implementation from _penalties.py_), averaged over the 20
last training epochs.

Datasets
--------

The code was implemented for 3 datasets:
- MNIST
- CIFAR10 (and upsampled CIFAR10, see paper Sec 4.2)
- a custom 12-class mini-ImageNet dataset (see below & paper Sec 4.2).

#### Getting our custom 12-class Mini-ImageNet Dataset
The custom mini-ImageNet was constructed by merging some similar classes together
from the usual ImageNet dataset (for example all the dogs). If the resulting
class got too big (e.g. the dog-class), we sampled a random subset of it.
The 12 classes obtained were:

> ball - car - cat - dog - elephant - fish - monkey - plane - ship -
> string-instrument - truck - wind-instrument

The list of images used in each class can be found in the imgnet12/ folder.  To
use our mini-imagenet dataloader, create a folder named _imgnet12/train/_
(resp. _imgnet12/val/_), with 12 subfolders (1 per class; name irrelevant)
containing the images listed in _train.txt_ (resp. _val.txt_).
