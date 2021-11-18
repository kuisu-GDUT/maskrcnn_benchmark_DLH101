#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：mask_rcnn_demo 
@File    ：Mask_RCNN_demo.py
@Author  ：kuisu
@Email     ：kuisu_dgut@163.com
@Date    ：2021/11/9 12:18 
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
# import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

#Those are the relevant imports for the detection model
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    # 权重文件路径
    parser.add_argument(
        "--config_file",
        default="./tools/config.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "--image_path",
        default=r'E:\Project_subject\DLH-101\defect detection\DLH_SSD300\data\002014.jpg',
        metavar="FILE",
        help="path to image file",
        type=str,
    )

    parser.add_argument(
        "--weight_path",
        default='./tools/model_final.pth',
        metavar="FILE",
        help="path to image file",
        type=str,
    )

    args = parser.parse_args()
    config_file = args.config_file

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.WEIGHT", args.weight_path])

    #Now we create the COCODemo object. It contains a few extra options for conveniency, such as the confidence threshold for detections to be shown.
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.2,
    )

    # from http://cocodataset.org/#explore?id=345434
    # image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
    image_path = args.image_path
    image = cv2.imread(image_path)

    imshow(image)

    #computing the predictions
    # compute predictions
    predictions = coco_demo.run_on_opencv_image(image)
    imshow(predictions)

if __name__ == '__main__':
    main()