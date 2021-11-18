# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from maskrcnn_benchmark.utils.registry import Registry

#定义带有注册功能字典的空字典
#Banckbone
BACKBONES = Registry()
RPN_HEADS = Registry()
#Box_head
ROI_BOX_FEATURE_EXTRACTORS = Registry()
ROI_BOX_PREDICTOR = Registry()
#keypoint_head
ROI_KEYPOINT_FEATURE_EXTRACTORS = Registry()
ROI_KEYPOINT_PREDICTOR = Registry()
#mask_head
ROI_MASK_FEATURE_EXTRACTORS = Registry()
ROI_MASK_PREDICTOR = Registry()
