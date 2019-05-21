import os
import sys
import argparse
import torch

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '..'))
import utils as ptutil
from configs import cfg
from model.model_zoo import get_model
from data.datasets.coco import COCODataset
from data.transforms.pre_process_cv import transforms_eval
from data.collate_batch import BatchCollator
from data.build import make_data_sampler, make_batch_data_sampler
from engine.inference import inference


def parse_args():
    parser = argparse.ArgumentParser(description='Eval RetinaNet.')
    parser.add_argument('--config_file', type=str,
                        default='../configs/retina_resnet50_v1b_coco.yaml')

    # device
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init-method', type=str, default="env://")

    args = parser.parse_args()
    return args


def get_dataset(name, transforms, root, ann_file):
    if name.lower() == 'coco':
        dataset = COCODataset(ann_file, root, False, transforms=transforms)
    else:
        raise ValueError('illegal dataset name')
    return dataset


def get_dataloader(cfg, distributed=False):
    aspect_grouping = [1]
    transform = transforms_eval(cfg.INPUT.min_size, cfg.INPUT.max_size)
    root = os.path.join(cfg.DATA.root, 'val2017')
    ann_file = os.path.join(cfg.DATA.root, 'annotations/instances_val2017.json')
    dataset = get_dataset(cfg.DATA.dataset, transform, root, ann_file)
    sampler = make_data_sampler(dataset, False, distributed)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, aspect_grouping, cfg.CONFIG.images_per_gpu, None, 0
    )
    collator = BatchCollator(cfg.CONFIG.size_divisibility)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.CONFIG.num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )
    return data_loader


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
    data_loader = get_dataloader(cfg, distributed)
    inference(
        model,
        data_loader,
        dataset_name=cfg.DATA.dataset,
        device=cfg.MODEL.device,
        expected_results=cfg.TEST.expected_results,
        expected_results_sigma_tol=cfg.TEST.expected_results_sigma_tol,
        output_folder=output_folder
    )
