# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import sys
import argparse

import torch

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '..'))
import utils as ptutil
from configs import cfg
from model.model_zoo import get_model
from data.build import build_dataloader
from solver import make_lr_scheduler, make_optimizer
from engine.training import training
from engine.inference import inference


def parse_args():
    parser = argparse.ArgumentParser(description='Eval RetinaNet.')
    parser.add_argument('--config-file', type=str,
                        default='../configs/retina_resnet50_v1s_coco.yaml')
    parser.add_argument("--skip-test", type=ptutil.str2bool, default='false',
                        help='Do not test the final model')

    # device
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")

    args = parser.parse_args()
    return args


def train(cfg, local_rank, distributed):
    pretrained_base = os.path.join(cfg.TRAIN.model_root, cfg.TRAIN.backbone+'.pth')
    model = get_model(cfg.MODEL.model, pretrained_base=pretrained_base)
    device = torch.device(cfg.MODEL.device)
    model.to(device)
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.CONFIG.output_dir

    save_to_disk = ptutil.get_rank() == 0
    checkpointer = ptutil.CheckPointer(
        model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.TRAIN.weight)
    arguments.update(extra_checkpoint_data)

    data_loader = build_dataloader(
        cfg, train=True, distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.TRAIN.checkpoint_period

    training(model, data_loader, optimizer, scheduler, checkpointer,
             device, checkpoint_period, arguments)

    return model


def evaluate(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    output_dir = cfg.CONFIG.output_dir
    if output_dir:
        output_folder = os.path.join(output_dir, "inference", cfg.DATA.dataset)
        ptutil.mkdir(output_folder)
    data_loader = build_dataloader(cfg, False, distributed)
    inference(
        model,
        data_loader,
        dataset_name=cfg.DATA.dataset,
        device=cfg.MODEL.device,
        expected_results=cfg.TEST.expected_results,
        expected_results_sigma_tol=cfg.TEST.expected_results_sigma_tol,
        output_folder=output_folder
    )


if __name__ == '__main__':
    args = parse_args()

    # init config
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.init_method)
        ptutil.synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_dir = cfg.CONFIG.output_dir
    if output_dir:
        ptutil.mkdir(output_dir)

    logger = ptutil.setup_logger("RetinaNet", output_dir, ptutil.get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        evaluate(cfg, model, args.distributed)
