来源于facebook开源的maskrcnn代码的复现, 并且作为demo给小组中的人使用.

代码结构
|--config 配置文件\

|--maskrcnn_benchmark\
|--|-- config\
|--|--|-- defaults.py #默认配置文件\
|--|--|-- paths_catalog.py #路径和模型配置文件\

|--|-- data\
|--|--|--datasets\
|--|--|--|-- evaluation\
|--|--|--|--|-- cityscapes #cityscapes的数据评估\
|--|--|--|--|--|-- cityscapes_eval.py #?\
|--|--|--|--|--|-- eval_instances.py #?\
|--|--|--|--|-- coco\
|--|--|--|--|--|-- abs_to_coco.py #将abs数据转为coco数据\
|--|--|--|--|--|-- coco_eval.py #评估coco的数据集\
|--|--|--|--|--|-- coco_eval_wrapper.py #?\
|--|--|--|--|--|-- coco_utils.py #获取coco数据的box, info等信息\
|--|--|--|--|-- voc\
|--|--|--|--|--|-- voc_eval.py # 评估voc的数据\
|--|--|--|-- abstract.py #自定义数据构建\
|--|--|--|-- cityscapes.py #cityscapes数据的构建\
|--|--|--|-- coco.py #加载coco数据集\
|--|--|--|-- concat_dataset.py #将多种数据进行组合\
|--|--|--|-- list_dataset.py #封装list数据集的组合\
|--|--|--|-- voc.py #读取voc数据集\
|--|--|--samplers\
|--|--|--|-- distributed.py # 实现分布数据训练功能\
|--|--|--|-- grouped_batch_sampler.py # 对数据根据长宽比进行分组\
|--|--|--|-- iteration_based_batch_sampler.py # 采样器, 根据迭代次数进行采样\
|--|--|--transforms\
|--|--|--|-- build.py # 构建数据处理的接口\
|--|--|--|-- transform.py #具体实现数据transform\
|--|--|--build.py #数据集构建\
|--|--|--collate_batch.py #自定义批量数据\

|--|-- engine\
|--|--|-- bbox_aug.py #?\
|--|--|-- inference.py #实现推理\
|--|--|-- trainer.py #实现训练\

|--|-- layers\
|--|--|-- _utils.py #加载c语言库和cuda库\
|--|--|-- batch_norm.py # 实现frozeBN\
|--|--|-- misc.py #实现空tensor的前后向传播\
|--|--|-- nms.py #实现nms功能->C加速\
|--|--|-- roi_align.py #实现roi_align功能\
|--|--|-- roi_pool.py #实现roi_pool功能\
|--|--|-- sigmoid_focal_loss.py #聚焦难负样本的训练\
|--|--|-- smooth_l1_loss.py #实现平滑损失\


|--|-- modeling\
|--|--|-- backbone \
|--|--|-- detector \
|--|--|-- roi_heads \
|--|--|-- rpn \
|--|--|-- balanced_positive_negative_sampler.py #实现难负样本采样\
|--|--|-- box_coder.py #对box进行编码和解码\
|--|--|-- make_layers.py #实现分组卷积和全连接等\
|--|--|-- matcher.py #实现anchor和target的匹配\
|--|--|-- poolers.py #实现将fpn层映射到对应的池化层上\
|--|--|-- registry.py #实现模型的注册功能\
|--|--|-- utils.py #模型构建工具\


|--|-- solver\
|--|--|-- build.py #构建训练器\ 
|--|--|-- lr_scheduler.py #构建lr衰减\


|--|-- structures\
|--|--|-- bounding_box.py #实现boxList\
|--|--|-- boxlist_ops.py #为boxlist\
|--|--|-- image_list.py #实现image_list类\
|--|--|-- keypoints.py #实现keypoints的数据处理\
|--|--|-- segmentation.py #实现对mask的处理\


|--|-- utils\
|--|--|-- c2_model_loading.py #将caffe2模型转为pytorch模型\
|--|--|-- checkpoint.py #实现模型加载和保存\
|--|--|-- collect_env.py #获取运行环境\
|--|--|-- comm.py #分布式的工具\
|--|--|-- cv2_utils.py #cv2的函数继承\
|--|--|-- env.py #运行环境设置?\
|--|--|-- imports.py #导入指定库文件\
|--|--|-- logger.py #日志功能\
|--|--|-- metric_logger.py #评估结果展示\
|--|--|-- miscellaneous.py #label,cfg等保存\
|--|--|-- model_serialization.py #根据权重名称加载模型权重\
|--|--|-- model_zoo.py #根据url下载模型权重\
|--|--|-- registry.py #实现模型得注册功能\
|--|--|-- timer.py #时间函数\

|-- tools\
|--|-- config.ymal #训练模型的参数\
|--|-- label2coco.py #将labelme标注的标签转为coco的label\
|--|-- labels.json # 标签名称以及对应的序号\
|--|-- log.txt #训练日志\
|--|-- PascalVOC2COCO.py #将pascalvoc转为coco的数据标签\
|--|-- train_net.py #训练模型\
|--|-- val_net.py #验证模型\

|-- Mask_RCNN_demo.py #指定一张图像, 进行检测\
|-- predictor.py #实现预测的方法\
|-- readme.md #说明文件\
|-- requirement.txt #环境要求\

argument
```args
#train_net.py
DATASETS.TRAIN
('DLH101_train',)
DATASETS.TEST
('DLH101_val',)
MODEL.ROI_BOX_HEAD.NUM_CLASSES
9

#val_net.py
MODEL.WEIGHT ./model_final.pth#指定模型位置
```