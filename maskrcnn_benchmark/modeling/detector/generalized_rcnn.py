# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone#构建backbone
from ..rpn.rpn import build_rpn#构建rpn
from ..roi_heads.roi_heads import build_roi_heads#构建roi_head


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.#利用前面网络输出的features和proposal来计算detections/masks
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)#根据配置信息创建backbone网络
        self.rpn = build_rpn(cfg, self.backbone.out_channels)#根据配置信息创建rpn
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)#根据配置信息创建roi_heads

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:#如果roi_heads不为none的话, 就直接计算其输出的结果
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:#训练模式下, 输出其损失值
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result#不在训练模式下, 则输出模型的预测结果
