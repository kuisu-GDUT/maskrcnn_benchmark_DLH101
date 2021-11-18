# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN


# 该函数是创建模型的入口函数，也是唯一的模型创建函数
_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}

def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]# 构建一个模型字典，虽然只有一对键值，但是方便后续的扩展
    return meta_arch(cfg)
    # 上面的语句等价于
    # return GeneralizedRCNN(cfg)
