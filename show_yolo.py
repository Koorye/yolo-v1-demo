#!/usr/bin/env python
# --*-- encoding: utf-8 --*--
# @Author: Koorye
# @Date: 2021-10-27
# @Desc: Yolo训练结果可视化

import cv2
import numpy as np
import os
import torch
import visdom
from tqdm import tqdm
from torchvision.transforms import transforms

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

trans = transforms.Compose([
    transforms.ToTensor(),
])

test_data = VOCDataset('test')


def draw_img_with_bbox(yolo, show_range, start_index=0, save=False):
    """
    绘制候选框的图片，用于查看可视化结果
    : param show_range: 展示范围，即展示的图片数量
    : param start_index: 要展示图片在数据集中的开始下标
    : param save: 是否保存
    : return: 返回 [b,w,h,c], [b,w,h,c] 预测和实际图片的张量
    """

    with torch.no_grad():
        yolo.eval()
        pbar = tqdm(range(start_index, show_range +
                    start_index), total=show_range)

        pred_imgs, target_imgs = [], []
        for i in pbar:
            data, target = test_data.__getitem__(i)
            data = data.to(device)

            output = yolo(data.unsqueeze(0))

            img = np.uint8(data.transpose(
                0, 1).transpose(1, 2).cpu().numpy() * 255)
            img2 = img.copy()
            output_bbox = labels2bbox(output.squeeze(0))
            target_bbox = labels2bbox(target)

            output_img = draw_bbox(img, output_bbox)
            target_img = draw_bbox(img2, target_bbox)

            if save:
                cv2.imwrite(os.path.join(OUTPUT_IMG_PATH,
                            f'pred{i}.jpg'), output_img)
                cv2.imwrite(os.path.join(OUTPUT_IMG_PATH,
                            f'target{i}.jpg'), target_img)

            b1, g1, r1 = cv2.split(output_img)
            b2, g2, r2 = cv2.split(target_img)
            output_img = cv2.merge((r1, g1, b1))
            target_img = cv2.merge((r2, g2, b2))
            pred_imgs.append(trans(output_img).numpy())
            target_imgs.append(trans(target_img).numpy())

        pred_imgs = torch.Tensor(np.array(pred_imgs))
        target_imgs = torch.Tensor(np.array(target_imgs))

        return pred_imgs, target_imgs


if __name__ == '__main__':
    viz = visdom.Visdom()
    pred_imgs, target_imgs = draw_img_with_bbox(yolo, 16, save=True)

    viz.images(pred_imgs, win='预测图片', opts={'title': '预测图片'})
    viz.images(target_imgs, win='实际图片', opts={'title': '实际图片'})
