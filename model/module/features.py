"""Feature extraction blocks.
Feature or Multi-Feature extraction is a key component in object detection.
Class predictor/Box predictor are usually applied on feature layer(s).
A good feature extraction mechanism is critical to performance.
"""
from __future__ import absolute_import

from torch import nn
import torch.nn.functional as F


def parse_network(network, outputs, pretrained, **kwargs):
    """Parse network with specified outputs and other arguments.

    Parameters
    ----------
    network : str or nn.Module
        Logic chain: load from gluoncv.model_zoo if network is string.
        Convert to Symbol if network is HybridBlock
    outputs : str or iterable of str
        The name of layers to be extracted as features.
    pretrained : bool
        Use pretrained parameters as in model_zoo

    Returns
    -------
    results: list of nn.Module (the same size as len(outputs))

    """
    l, n = len(outputs), len(outputs[0])
    results = [[] for _ in range(l)]
    if isinstance(network, str):
        from model.model_zoo import get_model
        network = get_model(network, pretrained=pretrained, **kwargs).features

    # helper func
    def recursive(pos, block, arr, j):
        if j == n:
            results[pos].append([block])
            return
        child = list(block.children())
        results[pos].append(child[:arr[j]])
        if pos + 1 < l: results[pos + 1].append(child[arr[j] + 1:])
        recursive(pos, child[arr[j]], arr, j + 1)

    block = list(network.children())

    for i in range(l):
        pos = outputs[i][0]
        if i == 0:
            results[i].append(block[:pos])
        elif i < l:
            results[i].append(block[outputs[i - 1][0] + 1: pos])
        recursive(i, block[pos], outputs[i], 1)

    for i in range(l):
        results[i] = nn.Sequential(*[item for sub in results[i] for item in sub if sub])
    return results


# top_blocks for FPN
class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)


class FPNFeatureExpander(nn.Module):
    """Feature extractor with additional layers to append.
    This is specified for ``Feature Pyramid Network for Object Detection``
    which implement ``Top-down pathway and lateral connections``.

    """

    def __init__(self, network, outputs, channels, num_filters, use_1x1=True, use_upsample=True,
                 use_elewadd=True, use_bias=False, use_relu=False, pretrained=False,
                 norm_layer=None, top_blocks=None):
        super(FPNFeatureExpander, self).__init__()
        self.features = nn.ModuleList(parse_network(network, outputs, pretrained))
        extras1 = [[] for _ in range(len(self.features))]
        extras2 = [[] for _ in range(len(self.features))]

        # num_filter is 256 in ori paper
        for i, (extra1, extra2, c, f) in enumerate(zip(extras1, extras2, channels, num_filters)):
            if use_1x1:
                extra1.append(nn.Conv2d(c, f, kernel_size=1, stride=1, bias=use_bias))
                if norm_layer is not None:
                    extra1.append(norm_layer(f))
            # Reduce the aliasing effect of upsampling described in ori paper
            extra2.append(nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1, bias=use_bias))
            if norm_layer is not None:
                extra2.append(norm_layer(f))
            if use_relu:
                extra2.append(nn.ReLU(inplace=True))
        self.extras1 = nn.ModuleList([nn.Sequential(*ext) for ext in extras1])
        self.extras2 = nn.ModuleList([nn.Sequential(*ext) for ext in extras2])
        self.top_blocks = top_blocks
        self.use_upsample, self.use_elewadd = use_upsample, use_elewadd

    def forward(self, x):
        feat_list = list()
        for feat in self.features:
            x = feat(x)
            feat_list.append(x)

        outputs, num = list(), len(feat_list)
        for i in range(num - 1, -1, -1):
            if i == num - 1:
                y = self.extras1[i](feat_list[i])
            else:
                bf = self.extras1[i](feat_list[i])
                if self.use_upsample:
                    y = F.interpolate(y, size=(bf.shape[2], bf.shape[3]), mode='nearest')
                if self.use_elewadd:
                    y = bf + y
            outputs.append(self.extras2[i](y))
        outputs = outputs[::-1]
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(feat_list[-1], outputs[-1])
            outputs.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(outputs[-1])
            outputs.extend(last_results)

        return outputs


if __name__ == '__main__':
    outputs = [[4, 2], [5, 3], [6, 5], [7, 2]]
    channels = [256, 512, 1024, 2048]
    num_filters = [256, 256, 256, 256]
    fpn = FPNFeatureExpander('resnet50_v1b', outputs, channels, num_filters, use_1x1=True, use_upsample=True,
                             use_elewadd=True, use_bias=False, use_relu=False, pretrained=False,
                             norm_layer=None, top_blocks=LastLevelP6P7)
    import torch

    a = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = fpn(a)
    print([a.shape for a in out])
