# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
from yacs.config import CfgNode as CN

_C = CN()

# MODEL
_C.MODEL = CN()
_C.MODEL.device = 'cuda'
_C.MODEL.model = ''

# INPUT
_C.INPUT = CN()
_C.INPUT.min_size_train = 800
_C.INPUT.max_size_train = 1333
_C.INPUT.min_size_test = 800
_C.INPUT.max_size_test = 1333

_C.INPUT.flip_prob = 0.5
_C.INPUT.brightness = 0.0
_C.INPUT.contrast = 0.0
_C.INPUT.saturation = 0.0
_C.INPUT.hue = 0.0

_C.INPUT.use_255 = True
_C.INPUT.pixel_mean = (102.9801, 115.9465, 122.7717)
_C.INPUT.pixel_std = (1.0, 1.0, 1.0)

# DATA
_C.DATA = CN()
_C.DATA.dataset = 'coco'
_C.DATA.root = os.path.expanduser('~/.torch/datasets/coco')

# TRAIN
_C.TRAIN = CN()
_C.TRAIN.model_root = os.path.expanduser('~/.torch/models')
_C.TRAIN.backbone = ''

_C.TRAIN.base_lr = 0.001
_C.TRAIN.bias_lr_factor = 2
_C.TRAIN.momentum = 0.9
_C.TRAIN.weight_decay = 0.0005
_C.TRAIN.weight_decay_bias = 0
_C.TRAIN.gamma = 0.1
_C.TRAIN.warmup_factor = 1.0 / 3
_C.TRAIN.warmup_iters = 500
_C.TRAIN.warmup_method = 'linear'

_C.TRAIN.max_iter = 45000
_C.TRAIN.steps = (30000,)
_C.TRAIN.weight = ''
_C.TRAIN.checkpoint_period = 2500

_C.TRAIN.img_per_gpu = 2

# TEST
_C.TEST = CN()
_C.TEST.expected_results = []
_C.TEST.expected_results_sigma_tol = 4

_C.TEST.pretrained = ''
_C.TEST.vis_thresh = 0.5
_C.TEST.img_per_gpu = 8

# CONFIG
_C.CONFIG = CN()
_C.CONFIG.size_divisibility = 32
_C.CONFIG.num_workers = 4

_C.CONFIG.save_dir = ''
_C.CONFIG.model = ''
_C.CONFIG.output_dir = '.'
