import os
import sys
import argparse
import matplotlib.pyplot as plt

import torch

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))
from model.model_zoo import get_model
from model.module import to_image_list
from model.ops import select_top_predictions
from data.transforms.pre_process_cv import load_test
from utils import str2bool
from utils.plot_bbox import plot_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Demo of RetinaNet.')
    parser.add_argument('--network', type=str, default='retina_resnet50_v1b_coco',
                        help="Base network name")
    parser.add_argument('--images', type=str, default=os.path.join(cur_path, '../png/biking.jpg'),
                        help='Test images.')
    parser.add_argument('--root', type=str, default=os.path.expanduser('~/.torch/models'),
                        help='Default pre-trained mdoel root.')
    parser.add_argument('--cuda', type=str2bool, default='True',
                        help='demo with GPU')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Threshold of object score when visualize the bboxes.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cpu')
    device_cpu = torch.device('cpu')
    if args.cuda:
        device = torch.device('cuda')
    image = args.images
    net = get_model(args.network, pretrained=True, root=args.root)
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
    top_predictions = select_top_predictions(prediction, conf_thresh=args.thresh)

    ax = plot_bbox(img, top_predictions, reverse_rgb=True,
                   class_names=net.classes, ax=ax)
    plt.show()
