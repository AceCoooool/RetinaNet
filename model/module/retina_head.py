import math
from torch import nn


class RetinaNetHead(nn.Module):
    """
    Adds a RetinaNet head with classification and regression heads
    """

    def __init__(self, in_channels, num_classes, aspect_ratios, scales_per_octave,
                 num_convs, prior_prob):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = num_classes - 1
        num_anchors = len(aspect_ratios) * scales_per_octave

        cls_tower = []
        bbox_tower = []
        for i in range(num_convs):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1))
            cls_tower.append(nn.ReLU(inplace=True))
            bbox_tower.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1))
            bbox_tower.append(nn.ReLU(inplace=True))

        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, 3, 1, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, 1, 1)

        # retinanet_bias_init
        self._weight_init()
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            logits.append(self.cls_logits(self.cls_tower(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
        return logits, bbox_reg
