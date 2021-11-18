# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.(实际上为stage2~5的最后一层)
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed-> 指定了送入fpn的每个feature map的通道数
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
                -> 当提供了 top_blocks时, 就会在fpn的最后输出上进行一个额外的op
                    然后result会扩展成result list返回
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        #假设我们使用的是resnet-50-fpn和配置, 则in_channels_list的值为:[256, 512, 1024, 2048]
        for idx, in_channels in enumerate(in_channels_list, 1):
            # 用下标起名: fpn_inner1, fpn_inner_2...
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            # 创建inner_block模块, 这里in_channels为各个stage输出的通道数
            # out_channels为256, 定义在用户配置文件中
            # 这里的卷积核大小为1, 其主要作用是改变通道数到out_channels (降维)
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            # 在当前特征图上添加fpn
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            # 将当前stage的fpn模块的名称添加到对应的列表中
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        # 将top_blocks作为FPN类的成员变量
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
            -> resnet的计算结果正好满足fpn的输入要求, 因此可以使用nn.Sequential直接将两者结合
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
                -> 经过fpn后的特征图组成的列表, 排列顺序是高分辨率的在前
        """
        # 先计算最后一层 (分辨率最低) 特征图的fpn结果
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        # [:-1]获取了前三项, [::-1]代表从头到尾切片, 步长为-1, 效果为列表逆置
        # 举例, zip里的操作self.inner_block[:-1][::-1]的运行结果为
        # [fpn_inner3, fpn_inner2, fpn_inner1], 相当于对列表进行逆置
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            # 根据给定的scale参数对特征图进行放大/缩小, 这里scale=2
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            # 将当前satage输出添加到结果列表中, 注意还要用layer_block执行卷积计算
            # 同时为了使得分辨率最大的在前,我们需要将结果插入到0位置.
            results.insert(0, getattr(self, layer_block)(last_inner))

        # 如果top_blocks不为空, 则需要执行如下额外的操作.
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)# 将新的计算结果追加进列表中.
        # 以元组(只读)形式返回
        return tuple(results)

# 最后一级的max pool层
class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    如果该模型采用retinanet需要采用多的p6和p7层.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
