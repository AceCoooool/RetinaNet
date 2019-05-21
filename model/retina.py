# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# copy from https://github.com/facebookresearch/maskrcnn-benchmark
import os
import torch
from torch import nn

from model.module import FPNFeatureExpander, LastLevelP6P7
from model.module import BoxCoder, to_image_list, make_anchor_generator
from model.module import RetinaNetHead, make_postprocessor, make_loss_evaluator

__all__ = ['get_retina_net', 'retina_resnet50_v1b_coco']


class RetinaNet(nn.Module):
    def __init__(self, features, classes, anchor_sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0),
                 anchor_strides=(8, 16, 32, 64, 128), straddle_thresh=-1, octave=2.0, scales_per_octave=3,
                 in_channels=256, num_convs=4, prior_prob=0.01, pre_nms_thresh=0.05, pre_nms_top_n=1000,
                 nms_thresh=0.4, fpn_post_nms_top_n=100, fg_iou_threshold=0.5, bg_iou_threshold=0.4,
                 loss_gamma=2.0, loss_alpha=0.25, bbox_reg_beta=0.11, bbox_reg_weight=4.0):
        super(RetinaNet, self).__init__()
        self.features = features
        self.classes = classes
        num_classes = len(classes) + 1
        anchor_generator = make_anchor_generator(
            anchor_sizes, aspect_ratios, anchor_strides, straddle_thresh, octave, scales_per_octave
        )
        head = RetinaNetHead(in_channels, num_classes, aspect_ratios, scales_per_octave,
                             num_convs, prior_prob)
        box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        box_selector_test = make_postprocessor(pre_nms_thresh, pre_nms_top_n, nms_thresh,
                                               fpn_post_nms_top_n, num_classes, box_coder)

        loss_evaluator = make_loss_evaluator(fg_iou_threshold, bg_iou_threshold, loss_gamma,
                                             loss_alpha, box_coder, bbox_reg_beta, bbox_reg_weight)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.features(images.tensors)
        box_cls, box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        # TODO: fix here
        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_train(self, anchors, box_cls, box_regression, targets):

        loss_box_cls, loss_box_reg = self.loss_evaluator(
            anchors, box_cls, box_regression, targets
        )
        losses = {
            "loss_retina_cls": loss_box_cls,
            "loss_retina_reg": loss_box_reg,
        }
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes


def get_retina_net(pretrained=None, **kwargs):
    net = RetinaNet(**kwargs)
    if pretrained:
        state = torch.load(pretrained)
        net.load_state_dict(state['model'] if 'model' in state else state)
    return net


def retina_resnet50_v1b_coco(pretrained=None, pretrained_base=None, **kwargs):
    from data.datasets import COCODataset
    classes = COCODataset.CLASSES
    pretrained_base = None if pretrained is None else pretrained_base
    features = FPNFeatureExpander(
        network='resnet50_v1b', outputs=[[5, 3], [6, 5], [7, 2]],
        channels=[512, 1024, 2048], num_filters=[256, 256, 256],
        use_1x1=True, use_upsample=True, use_elewadd=True, use_bias=True,
        pretrained=pretrained_base, top_blocks=LastLevelP6P7(2048, 256))
    return get_retina_net(
        pretrained=pretrained, features=features,
        classes=classes, anchor_sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32, 64, 128), straddle_thresh=-1, octave=2.0, scales_per_octave=3,
        in_channels=256, num_convs=4, prior_prob=0.01, pre_nms_thresh=0.05, pre_nms_top_n=1000,
        nms_thresh=0.4, fpn_post_nms_top_n=100, fg_iou_threshold=0.5, bg_iou_threshold=0.4,
        loss_gamma=2.0, loss_alpha=0.25, bbox_reg_beta=0.11, bbox_reg_weight=4.0, **kwargs)


if __name__ == '__main__':
    a = ''
    if a:
        print('a')

    # net = retina_resnet50_v1b_coco(pretrained=True)
    # net.eval()
    # params = net.state_dict()
    # all_keys = params.keys()
    # all_keys = [k for k in all_keys if not k.endswith('num_batches_tracked')]
    # for k in all_keys:
    #     print(":\"" + k+"\",")
    # print(len(all_keys))
