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
from engine.inference import inference


def parse_args():
    parser = argparse.ArgumentParser(description='Eval RetinaNet.')
    parser.add_argument('--config-file', type=str,
                        default='../configs/retina_resnet101_v1b_coco.yaml')

    # device
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # init config
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.init_method)
        ptutil.synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # logging
    logger = ptutil.setup_logger("RetinaNet", cfg.CONFIG.save_dir, ptutil.get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    model = get_model(cfg.CONFIG.model, pretrained=cfg.TEST.pretrained)
    model.to(cfg.MODEL.device)

    output_dir = cfg.CONFIG.output_dir
    if output_dir:
        output_folder = os.path.join(output_dir, "inference", cfg.DATA.dataset)
        ptutil.mkdir(output_folder)

    # dataset
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
