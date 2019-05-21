# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
from yacs.config import CfgNode as CN

_C = CN()

# MODEL
_C.MODEL = CN()
_C.MODEL.device = 'cuda'

# INPUT
_C.INPUT = CN()
_C.INPUT.min_size = 800
_C.INPUT.max_size = 1333

# DATA
_C.DATA = CN()
_C.DATA.dataset = 'coco'
_C.DATA.root = os.path.expanduser('~/.torch/datasets/coco')

# TEST
_C.TEST = CN()
_C.TEST.expected_results = []
_C.TEST.expected_results_sigma_tol = 4

_C.TEST.pretrained = ''


# CONFIG
_C.CONFIG = CN()
_C.CONFIG.size_divisibility = 32
_C.CONFIG.num_workers = 4

_C.CONFIG.save_dir = ''
_C.CONFIG.model = ''
_C.CONFIG.output_dir = '.'
_C.CONFIG.images_per_gpu = 1
