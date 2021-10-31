#!/usr/bin/env python
# --*-- encoding: utf-8 --*--
# @Author: Koorye
# @Date: 2021-10-27
# @Desc: 数据预处理，制作重设尺寸并填充的图片和标签CSV文件

import cv2
import numpy as np
import os
import pandas as pd
import random
import shutil
from tqdm import tqdm

# 类别列表
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

# DATA_PATH: 要处理的数据集目录
# OUTPUT_PATH: 输出的根目录
# OUTPUT_IMG_PATH: 输出的图片目录
# OUTPUT_LABEL_PATH: 输出的标签目录
DATA_PATH = 'data/VOC2012'
OUTPUT_PATH = 'data'
OUTPUT_IMG_PATH = 'data/img'
OUTPUT_LABEL_PATH = 'data/label'

def convert(size, box):
    """
    计算归一化后的x,y,w,h
    : param size: (w,h) 宽度和高度信息
    : param box:  (x1,y1,x2,y2) 包含左上角坐标(x1,y1)和右下角坐标(x2,y2)
    : return: x,y,w,h 中心点的相对坐标和宽高
    """

    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(annotation_file):
    """
    将XML注释文件转换为目标检测的CSV标签文件
    其中包含物体的类别，bbox的左上角点坐标以及bbox的宽、高
    并将四个物理量归一化
    : param annotation_file: XML文件
    """

    image_id = annotation_file.split('.')[0]
    annotation_file = os.path.join(DATA_PATH, 'Annotations', annotation_file)

    size_df = pd.read_xml(
        annotation_file, xpath='.//size')[['width', 'height']]
    size = (size_df['width'].tolist()[0], size_df['height'].tolist()[0])

    name_df = pd.read_xml(annotation_file, xpath='.//object')[['name']]
    pos_df = pd.read_xml(annotation_file, xpath='.//object/bndbox')
    df = pd.concat([name_df, pos_df], axis=1)[['name', 'xmin', 'ymin', 'xmax', 'ymax']]

    df['new'] = df.apply(lambda row: convert(size, row.tolist()[1:]), axis=1)
    df[['x','y','w','h']] = df['new'].apply(pd.Series)

    df = df[df['name'].apply(lambda x: x in CLASSES)]
    df['id'] = df['name'].apply(lambda x: CLASSES.index(x)).astype(int)
    df = df[['id', 'x', 'y', 'w', 'h']]

    df.to_csv(os.path.join(OUTPUT_LABEL_PATH, f'{image_id}.csv'), index=None)


def generate_label():
    """
    遍历Annotation目录中的所有{image_id}.xml文件
    生成{image_df}.csv的数据文件，包含以下列
    : param name: 物体名称
    : param x: 中心横坐标
    : param y: 中心纵坐标
    : param w: 宽度
    : param h: 高度
    """

    filenames = os.listdir(os.path.join(DATA_PATH, 'Annotations'))
    pbar = tqdm(filenames, total=len(filenames))
    for file in pbar:
        convert_annotation(file)

def padding_resize_img(img_name, img_size):
    """
    填充并修改标签尺寸
    : param img_name: 图片名(包含后缀)
    : param img_size: 重设后的图片尺寸 img_size x img_size
    """

    img_name_ = img_name.split('.')[0]
    img = cv2.imread(os.path.join(DATA_PATH, 'JPEGImages', img_name))
    h, w = img.shape[:2]
    padw, padh = 0, 0
    if h > w:
        padw = (h-w) // 2
        img = np.pad(img,((0,0),(padw,padw),(0,0)),'constant',constant_values=0)
    elif w > h:
        padh = (w - h) // 2
        img = np.pad(img,((padh,padh),(0,0),(0,0)), 'constant', constant_values=0)

    df = pd.read_csv(os.path.join(OUTPUT_LABEL_PATH, f'{img_name_}.csv'))    
    if padw != 0:
        df['x'] = (df['x'] * w + padw) / h
        df['w'] = (df['w'] * w) / h
    elif padh != 0:
        df['y'] = (df['y'] * h + padh) / w
        df['h'] = (df['h'] * h) / w
    df.to_csv(os.path.join(OUTPUT_LABEL_PATH, f'{img_name_}.csv'), index=None)

    img = cv2.resize(img, (img_size, img_size))
    cv2.imwrite(os.path.join(OUTPUT_IMG_PATH, img_name), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

def generate_img():
    """
    修改所有图片并修正标签
    """

    filenames = os.listdir(os.path.join(DATA_PATH, 'JPEGImages'))
    pbar = tqdm(filenames, total=len(filenames))
    for file in pbar:
        padding_resize_img(file, 448)

def copy_train_and_test():
    """
    将train和test文件复制到数据目录下
    """

    copy_root = os.path.join(DATA_PATH, 'ImageSets', 'Main')
    shutil.copyfile(os.path.join(copy_root, 'train.txt'), os.path.join(OUTPUT_IMG_PATH, 'train.txt'))
    shutil.copyfile(os.path.join(copy_root, 'val.txt'), os.path.join(OUTPUT_IMG_PATH, 'val.txt'))
    shutil.copyfile(os.path.join(copy_root, 'trainval.txt'), os.path.join(OUTPUT_IMG_PATH, 'trainval.txt'))

def train_test_split(train_rate=.8, seed=None):
    """
    根据所有图片随机切分训练集和测试集
    """
    
    if seed is not None:
        random.seed(seed)
    files = [x.split('.')[0] for x in os.listdir(OUTPUT_IMG_PATH)]
    train_files = random.sample(files, int(train_rate*len(files)))
    test_files = [x for x in files if x not in train_files]
    with open(os.path.join(OUTPUT_PATH, 'train.txt'), 'w') as f:
        pbar = tqdm(train_files, total=len(train_files), desc='生成训练文件')
        for file in pbar:
            f.write(file+'\n')
    with open(os.path.join(OUTPUT_PATH, 'test.txt'), 'w') as f:
        pbar = tqdm(test_files, total=len(test_files), desc='生成测试文件')
        for file in pbar:
            f.write(file+'\n')

if __name__ == '__main__':
    generate_label()
    generate_img()
    # copy_train_and_test()
    train_test_split()
