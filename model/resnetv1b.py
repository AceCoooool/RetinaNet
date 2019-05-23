"""ResNetV1bs, implemented in PyTorch."""
import os
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['resnet50_v1b', 'resnet101_v1b',
           'resnet50_v1s']


# -----------------------------------------------------------------------------
# BLOCKS & BOTTLENECK
# -----------------------------------------------------------------------------
class BasicBlockV1b(nn.Module):
    """ResNetV1b BasicBlockV1b"""
    expansion = 1

    def __init__(self, in_channel, planes, strides=1, dilation=1, downsample=None,
                 previous_dilation=1, **kwargs):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, planes, kernel_size=3, stride=strides,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.strides = strides

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu2(out)

        return out


# This is old bottleneck version: stride on first conv
class BottleneckV1(nn.Module):
    """ResNetV1 BottleneckV1b"""
    expansion = 4

    def __init__(self, in_channel, planes, strides=1, dilation=1,
                 downsample=None, last_gamma=False, **kwargs):
        super(BottleneckV1, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, planes, kernel_size=1, stride=strides, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if not last_gamma:
            self.bn3 = nn.BatchNorm2d(planes * 4)
        else:
            self.bn3 = nn.BatchNorm2d(planes * 4)
            nn.init.zeros_(self.bn3.weight)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu3(out)

        return out


class BottleneckV1b(nn.Module):
    """ResNetV1b BottleneckV1b"""
    expansion = 4

    def __init__(self, in_channel, planes, strides=1, dilation=1,
                 downsample=None, last_gamma=False, **kwargs):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=strides,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if not last_gamma:
            self.bn3 = nn.BatchNorm2d(planes * 4)
        else:
            self.bn3 = nn.BatchNorm2d(planes * 4)
            nn.init.zeros_(self.bn3.weight)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class ResNetV1b(nn.Module):
    """ Pre-trained ResNetV1b Model, which produces the strides of 8
    feature maps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.


    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, classes=1000, dilated=False, last_gamma=False, deep_stem=False,
                 stem_width=32, avg_down=False, final_drop=0.0):
        channel = [64, 64, 128, 256] if block is BasicBlockV1b else [64, 256, 512, 1024]
        self.basic = block is BasicBlockV1b
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNetV1b, self).__init__()
        self.features = list()
        if not deep_stem:
            self.features.append(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        else:
            self.features.append(nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False))
            self.features.append(nn.BatchNorm2d(stem_width))
            self.features.append(nn.ReLU(inplace=True))
            self.features.append(nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False))
            self.features.append(nn.BatchNorm2d(stem_width))
            self.features.append(nn.ReLU(inplace=True))
            self.features.append(nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                                           padding=1, bias=False))
            channel[0] = stem_width * 2
        self.features.append(nn.BatchNorm2d(stem_width * 2))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.features.append(self._make_layer(block, channel[0], 64, layers[0], avg_down=avg_down,
                                              last_gamma=last_gamma))
        self.features.append(self._make_layer(block, channel[1], 128, layers[1], strides=2, avg_down=avg_down,
                                              last_gamma=last_gamma))
        if dilated:
            self.features.append(self._make_layer(block, channel[2], 256, layers[2], strides=1, dilation=2,
                                                  avg_down=avg_down, last_gamma=last_gamma))
            self.features.append(self._make_layer(block, channel[3], 512, layers[3], strides=1, dilation=4,
                                                  avg_down=avg_down, last_gamma=last_gamma))
        else:
            self.features.append(self._make_layer(block, channel[2], 256, layers[2], strides=2,
                                                  avg_down=avg_down, last_gamma=last_gamma))
            self.features.append(self._make_layer(block, channel[3], 512, layers[3], strides=2,
                                                  avg_down=avg_down, last_gamma=last_gamma))
        self.features = nn.Sequential(*self.features)
        self.drop = None
        if final_drop > 0.0:
            self.drop = nn.Dropout(final_drop)
        self.fc = nn.Linear(512 * block.expansion, classes)

    def _make_layer(self, block, in_channel, planes, blocks, strides=1, dilation=1,
                    avg_down=False, last_gamma=False):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = list()
            if avg_down:
                if dilation == 1:
                    downsample.append(nn.AvgPool2d(kernel_size=strides, stride=strides,
                                                   ceil_mode=True, count_include_pad=False))
                else:
                    downsample.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                   ceil_mode=True, count_include_pad=False))
                downsample.append(nn.Conv2d(in_channel, planes * block.expansion, kernel_size=1,
                                            stride=1, bias=False))
                downsample.append(nn.BatchNorm2d(planes * block.expansion))
            else:
                downsample.append(nn.Conv2d(in_channel, planes * block.expansion,
                                            kernel_size=1, stride=strides, bias=False))
                downsample.append(nn.BatchNorm2d(planes * block.expansion))
            downsample = nn.Sequential(*downsample)
        layers = list()
        if dilation in (1, 2):
            layers.append(block(in_channel, planes, strides, dilation=1,
                                downsample=downsample, previous_dilation=dilation,
                                last_gamma=last_gamma))
        elif dilation == 4:
            layers.append(block(in_channel, planes, strides, dilation=2,
                                downsample=downsample, previous_dilation=dilation,
                                last_gamma=last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes if self.basic else planes * 4, planes, dilation=dilation,
                                previous_dilation=dilation, last_gamma=last_gamma))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)

        x = F.adaptive_avg_pool2d(x, 1).squeeze(3).squeeze(2)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def resnet50_v1b(pretrained=None, **kwargs):
    model = ResNetV1b(BottleneckV1, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
        from data.datasets.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet101_v1b(pretrained=None, **kwargs):
    model = ResNetV1b(BottleneckV1, [3, 4, 23, 3], **kwargs)
    if pretrained:
        import torch
        model.load_state_dict(torch.load(pretrained))
        from data.datasets.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1s(pretrained=None, **kwargs):
    """Constructs a ResNetV1s-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        import torch
        model.load_state_dict(torch.load(pretrained))
        from data.datasets.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


if __name__ == '__main__':
    net = resnet101_v1b()
    print(net)
