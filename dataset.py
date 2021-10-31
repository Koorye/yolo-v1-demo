#!/usr/bin/env python
# --*-- encoding: utf-8 --*--
# @Author: Koorye
# @Date: 2021-10-27
# @Desc: VOC 2012数据集

import cv2
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, dataset
from torchvision.transforms import transforms

# DATA_PATH: 数据集的根路径
# IMG_PATH: 图片的保存路径
# LABEL_PATH: 标签的保存路径
DATA_PATH = 'data'
IMG_PATH = 'data/img'
LABEL_PATH = 'data/label'

CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']


def convert_bbox2labels(bbox_df):
    """
    将bbox的DataFrame转换为[7,7,30]的格式
    : param bbox_df: bbox的DataFrame -> [id,x,y,w,h]
    : return: [7,7,30]
    """

    nrow, ncol = 7, 7
    label = np.zeros((7, 7, 30))

    df = bbox_df.copy()

    # 计算每个坐标落在网格的行和列
    df['col'] = df['x'] * ncol
    df['row'] = df['y'] * nrow
    df['col'] = df['col'].astype(int)
    df['row'] = df['row'].astype(int)

    # 计算相对坐标
    df['px'] = df['x'] * ncol - df['col']
    df['py'] = df['y'] * nrow - df['row']

    # 排除多个物体在同一网格的情况
    historical_pos = []
    for _, row in df.iterrows():
        pos = (int(row['row']), int(row['col']))
        if pos not in historical_pos:
            historical_pos.append(pos)
        else:
            continue

        label[int(row['row']), int(row['col']), 0:5] = np.array(
            [row['px'], row['py'], row['w'], row['h'], 1])
        label[int(row['row']), int(row['col']), 5:10] = np.array(
            [row['px'], row['py'], row['w'], row['h'], 1])
        label[int(row['row']), int(row['col']), 10+int(row['id'])] = 1

    return label


class VOCDataset(Dataset):
    """
    VOC 2012数据集
    """

    def __init__(self, mode='train') -> None:
        super(VOCDataset, self).__init__()

        self.files = []
        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        with open(os.path.join(DATA_PATH, f'{mode}.txt'), 'r') as f:
            self.files = [x.strip() for x in f]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(IMG_PATH, f'{self.files[index]}.jpg'))
        # 读取CSV文件，转换为[7,7,30]的格式
        df = pd.read_csv(os.path.join(LABEL_PATH, f'{self.files[index]}.csv'))
        return self.trans(img), torch.FloatTensor(convert_bbox2labels(df))


if __name__ == '__main__':
    # 以下代码实现输入核的可视化，以便校验
    dataset = VOCDataset(mode='test')
    # index可任意设置
    img, label = dataset.__getitem__(6)
    print(img.size())
    print(label.size())

    img = img * 255
    img = img.cpu().transpose(0, 1).transpose(1, 2).numpy().astype(np.uint8)

    cell_size = 448 // 7
    for i in range(8):
        if i*cell_size < 448:
            img[int(i*cell_size), :, :] = 0
            img[:, int(i*cell_size), :] = 0

    x = label[:, :, 0]
    y = label[:, :, 1]
    w = label[:, :, 2]
    h = label[:, :, 3]
    obj = label[:, :, 10:].sum(axis=2)
    for row in range(len(obj)):
        for col in range(len(obj[0])):
            if obj[row, col] != 0:
                posx = int((x[row, col]+col)*448/7)
                posy = int((y[row, col]+row)*448/7)
                width = int(w[row, col]*448)
                height = int(h[row, col]*448)

                img = cv2.circle(img, (posx, posy), 5, (0, 0, 225), -1)
                img = cv2.rectangle(img, (int(posx-width/2), int(posy-height/2)),
                                    (int(posx+width/2), int(posy+height/2)), (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
