#!/usr/bin/env python
# --*-- encoding: utf-8 --*--
# @Author: Koorye
# @Date: 2021-10-27
# @Desc: Yolo训练结果可视化

import cv2
import numpy as np
import os
import torch
from tqdm import tqdm

from util import draw_bbox, labels2bbox, load_model
from dataset import VOCDataset

# OUTPUT_MODEL_PATH: 输出模型路径
# OUTPUT_IMG_PATH: 输出图片路径
# HISTORICAL_EPOCHS: 历史训练次数
OUTPUT_MODEL_PATH = 'output/model'
OUTPUT_IMG_PATH = 'output/img'
HISTORICAL_EPOCHS = -1

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo, _ = load_model(HISTORICAL_EPOCHS, device)

test_data = VOCDataset('test')
def save_img_with_bbox(show_range, start_index=0):
    """
    保存绘制候选框的图片，用于查看可视化结果
    : param show_range: 展示范围，即展示的图片数量
    : param start_index: 要展示图片在数据集中的开始下标
    """

    with torch.no_grad():
        yolo.eval()
        pbar = tqdm(range(start_index, show_range+start_index), total=show_range)
        for i in pbar:
            data, target = test_data.__getitem__(i)
            data = data.to(device)

            output = yolo(data.unsqueeze(0))

            img = np.uint8(data.transpose(0,1).transpose(1,2).cpu().numpy() * 255)
            img2 = img.copy()
            output_bbox = labels2bbox(output.squeeze(0))
            target_bbox = labels2bbox(target)

            output_img = draw_bbox(img, output_bbox)
            target_img = draw_bbox(img2, target_bbox)
            cv2.imwrite(os.path.join(OUTPUT_IMG_PATH, f'pred{i}.jpg'), output_img)
            cv2.imwrite(os.path.join(OUTPUT_IMG_PATH, f'target{i}.jpg'), target_img)

if __name__ == '__main__':
    save_img_with_bbox(10)