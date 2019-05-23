import os
from model.resnetv1b import *
from model.retina import *

__all__ = ['get_model', 'get_model_list']

_models = {
    # backbone
    'resnet50_v1b': resnet50_v1b,
    'resnet101_v1b': resnet101_v1b,
    'resnet50_v1s': resnet50_v1s,
    # RetinaNet
    'retina_resnet50_v1b_coco': retina_resnet50_v1b_coco,
    'retina_resnet101_v1b_coco': retina_resnet101_v1b_coco,
    'retina_resnet50_v1s_coco': retina_resnet50_v1s_coco,

}


def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % name
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


def get_model_list():
    """Get the entire list of model names in model_zoo.

    Returns
    -------
    list of str
        Entire list of model names in model_zoo.

    """
    return _models.keys()


def get_model_file(name, root=os.path.expanduser('~/.torch/models')):
    file_path = os.path.join(root, name + '.pth')
    if os.path.exists(file_path):
        return file_path
    else:
        raise ValueError('Model file is not found. Please convert it first.')
