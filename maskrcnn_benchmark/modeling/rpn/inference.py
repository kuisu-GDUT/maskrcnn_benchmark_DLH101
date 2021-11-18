# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
from .utils import permute_and_flatten

class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    主要完成对RPN box的后处理功能 (在将boxes送到heads之前执行)
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
        fpn_post_nms_per_batch=True,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        #创建一个BoxCoder实例
        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.fpn_post_nms_per_batch = fpn_post_nms_per_batch

    def add_gt_proposals(self, proposals, targets):
        """
        将真实的边框标签targets添加到BoxList列表数据中
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        #将target进行深度复制, gt_boxes是一个列表, 其元素类型为BoxList
        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))
        #调用boxlist_ops.py中的cat_boxlist函数将proposal和gt_box合成一个boxlist
        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        在单一的特征图上执行前向传播
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        # put in the same format as anchors
        # 将格式转换成和anchors相同的格式, 想改变维度的排列, 然后改变shape的形状
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        #sigmoid归一化
        objectness = objectness.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        #调用pytorch的topk函数, 该函数返回两个列表, 一个topk的值, 一个是对应下标
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        #创建batch的下标, shape为Nx1, 按照顺序递增
        batch_idx = torch.arange(N, device=device)[:, None]
        #获取所有的batch的top_k box
        box_regression = box_regression[batch_idx, topk_idx]

        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 4)

        result = []
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            #1. 根据当前的结果创建boxlist实例
            boxlist = BoxList(proposal, im_shape, mode="xyxy")
            #2. 添加score
            boxlist.add_field("objectness", score)
            #3. 防止box超出image的边界
            boxlist = boxlist.clip_to_image(remove_empty=False)
            #4. 移除过小的box
            boxlist = remove_small_boxes(boxlist, self.min_size)
            #5. 在当前的box上执行nms算法
            boxlist = boxlist_nms(
                boxlist,
                self.nms_thresh,
                max_proposals=self.post_nms_top_n,
                score_field="objectness",
            )
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
                经过box decoding和NMS操作处理后的anchors
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        # 添加gt bboxes到proposal当中去
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists

    #在所有层次上进行选择
    def select_over_all_levels(self, boxlists):
        #在训练阶段和测试阶段的行为不同, 在训练阶段, post_nms_top_n是在所有的proposals上进行的
        #而在测试阶段, 是在每一个图片的proposal上进行.
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # NOTE: it should be per image, and not per batch. However, to be consistent 
        # with Detectron, the default is per batch (see Issue #672)
        if self.training and self.fpn_post_nms_per_batch:
            #拼接 'objectness'
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            #获取box数量
            box_sizes = [len(boxlist) for boxlist in boxlists]
            #防止post_nms_top_n超过anchors总数, 产生错误
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            #获取topk的下标
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.bool)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            # 获取所有满足条件的box
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN#eg: 2000
    if not is_train:# 1000
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    fpn_post_nms_per_batch = config.MODEL.RPN.FPN_POST_NMS_PER_BATCH
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    #根据配置参数, 创建一个RPNPostProcessor实例
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        fpn_post_nms_per_batch=fpn_post_nms_per_batch,
    )
    return box_selector
