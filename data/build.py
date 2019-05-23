import os
import bisect
import copy

from torch.utils import data

from data.collate_batch import BatchCollator
from data.datasets.coco import COCODataset
from data.samplers import DistributedSampler, GroupedBatchSampler, IterationBasedBatchSampler
from data.transforms.pre_process import transforms_eval, transforms_train


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = data.sampler.RandomSampler(dataset)
    else:
        sampler = data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(dataset, sampler, aspect_grouping, batch_size, num_iters=None, start_iter=0):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = GroupedBatchSampler(
            sampler, group_ids, batch_size, drop_uneven=False
        )
    else:
        batch_sampler = data.sampler.BatchSampler(
            sampler, batch_size, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def build_dataset(name, transforms, root, ann_file, remove):
    if name.lower() == 'coco':
        dataset = COCODataset(ann_file, root, remove, transforms=transforms)
    else:
        raise ValueError('illegal dataset name')
    return dataset


def build_dataloader(cfg, train=False, distributed=False, start_iter=0):
    aspect_grouping = [1]
    if train:
        transform = transforms_train(cfg)
        num_iters = cfg.TRAIN.max_iter
        shuffle, remove = True, True
        img_per_gpu = cfg.TRAIN.img_per_gpu
        root = os.path.join(cfg.DATA.root, 'train2017')
        ann_file = os.path.join(cfg.DATA.root, 'annotations/instances_train2017.json')
    else:
        transform = transforms_eval(cfg)
        num_iters, start_iter = None, 0
        shuffle, remove = False, False
        img_per_gpu = cfg.TEST.img_per_gpu
        root = os.path.join(cfg.DATA.root, 'val2017')
        ann_file = os.path.join(cfg.DATA.root, 'annotations/instances_val2017.json')
    dataset = build_dataset(cfg.DATA.dataset, transform, root, ann_file, remove)
    sampler = make_data_sampler(dataset, shuffle, distributed)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, aspect_grouping, img_per_gpu, num_iters, start_iter
    )
    collator = BatchCollator(cfg.CONFIG.size_divisibility)
    data_loader = data.DataLoader(
        dataset,
        num_workers=cfg.CONFIG.num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )
    return data_loader
