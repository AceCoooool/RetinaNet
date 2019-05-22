import os
import sys
import argparse
import matplotlib.pyplot as plt

import torch

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from configs import cfg
from model.model_zoo import get_model
from model.module import to_image_list
from model.ops import select_top_predictions
from data.transforms.pre_process import load_test
from utils.plot_bbox import plot_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Demo of RetinaNet.')
    parser.add_argument('--config_file', type=str,
                        default='../configs/retina_resnet50_v1b_coco.yaml')
    parser.add_argument('--images', type=str, default=os.path.join(cur_path, '../png/biking.jpg'),
                        help='Test images.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    device = torch.device(cfg.MODEL.device)
    device_cpu = torch.device('cpu')
    image = args.images
    net = get_model(cfg.CONFIG.model, pretrained=cfg.TEST.pretrained)
    net.to(device)
    net.eval()

    ax = None
    x, img = load_test(image, min_image_size=800)
    x = to_image_list(x, 32)
    x = x.to(device)
    with torch.no_grad():
        predictions = net(x)
    predictions = [o.to(device_cpu) for o in predictions]
    prediction = predictions[0]
    height, width = img.shape[:-1]
    prediction = prediction.resize((width, height))
    top_predictions = select_top_predictions(prediction, conf_thresh=cfg.TEST.vis_thresh)

    ax = plot_bbox(img, top_predictions, reverse_rgb=True,
                   class_names=net.classes, ax=ax)
    plt.show()
