# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference

# from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,#学习更新策略, 封装在solver/lr_scheduler.py
    checkpointer,#DetectronCheckpointer, 用于自动转化caffe2 Detectron的模型文件
    device,
    checkpoint_period,#指定模型的保存迭代间隔
    test_period,
    arguments,#额外的其他参数, 字典类型, 一般情况只有arguments[iteratioin], 初值为0
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")#记录日志信息
    logger.info("Start training")
    #用于记录一些变量的滑动平均值和全局平均值
    meters = MetricLogger(delimiter="  ")#delimiter为定界符,

    #数据载入器重写了len函数, 使其返回载入器需要提供的batch的次数, 即cfg.SOLVER.MAX_ITER
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]#默认为0, 但是会根据载入的权重文件, 变成其他值.
    model.train()
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST

    loss_list = []
    lr_list = []

    #遍历data_loader, 第二个参数是设置序号的开始序号.
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end#获取一个batch所需的时间
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)#根据image和targets计算loss

        losses = sum(loss for loss in loss_dict.values())#将各个loss合并

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)#更新滑动平均值

        optimizer.zero_grad()#清除梯度缓存
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # with amp.scale_loss(losses, optimizer) as scaled_losses:
        #     scaled_losses.backward()#todo 被注释掉
        losses.backward()#计算梯度
        optimizer.step()#更新参数
        scheduler.step()#更新一次学习率

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        #根据时间的滑动平均值计算大约还剩多长时间结束训练
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        #每经过20此迭代, 输出一次训练状态
        if iteration % 20 == 0 or iteration == max_iter:
            loss_list.append(losses_reduced.item())
            lr_list.append(optimizer.param_groups[0]["lr"])
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "lr: {lr:.6f}",
                        "{meters}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    lr=optimizer.param_groups[0]["lr"],
                    meters=str(meters),
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        #模型保存
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

        #验证集验证
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
            )
            synchronize()
            model.train()
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "lr: {lr:.6f}",
                        "{meters}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        #达到最大迭代次数后, 也进行保存.
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    #输出总的训练耗时
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    print(loss_list,lr_list)
    # from plot_curve import plot_loss_and_lr
    # with open('loss_result.txt','w') as f:
    #     for i in range(len(loss_list)):
    #         f.write("{} {} {} \n".format(i,loss_list[i],lr_list[i]))
    #
    # plot_loss_and_lr(loss_list,lr_list)
