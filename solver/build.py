# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from solver.lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.TRAIN.base_lr
        weight_decay = cfg.TRAIN.weight_decay
        if "bias" in key:
            lr = cfg.TRAIN.base_lr * cfg.TRAIN.bias_lr_factor
            weight_decay = cfg.TRAIN.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.TRAIN.momentum)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.TRAIN.steps,
        cfg.TRAIN.gamma,
        warmup_factor=cfg.TRAIN.warmup_factor,
        warmup_iters=cfg.TRAIN.warmup_iters,
        warmup_method=cfg.TRAIN.warmup_method,
    )
