# encoding: utf-8

from .baseline import Baseline
from reid.modeling.backbones.resnet import *

def build_model(cfg, num_classes):

    model = Baseline(num_classes, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.LAST_STRIDE)
    return model

__factory = {
    'resnet50': ResNet,

}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
