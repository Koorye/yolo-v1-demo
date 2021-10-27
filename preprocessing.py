import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

DATA_PATH = 'data/VOC2012'
OUTPUT_IMG_PATH = 'data/img'
OUTPUT_LABEL_PATH = 'data/label'
# DATA_PATH = 'example'
# OUTPUT_IMG_PATH = 'example'
# OUTPUT_LABEL_PATH = 'example'

def convert(size, box):
    """
    计算归一化后的x,y,w,h
    size: (w,h) 宽度和高度信息
    box:  (x1,y1,x2,y2) 包含左上角坐标(x1,y1)和右下角坐标(x2,y2)
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
    把图像image_id的xml文件转换为目标检测的label文件(csv)
    其中包含物体的类别，bbox的左上角点坐标以及bbox的宽、高
    并将四个物理量归一化
    """

    image_id = annotation_file.split('.')[0]
    annotation_file = os.path.join(DATA_PATH, 'Annotations', annotation_file)

    size_df = pd.read_xml(
        annotation_file, xpath='.//size')[['width', 'height']]
    size = (size_df['width'].tolist()[0], size_df['height'].tolist()[0])

    name_df = pd.read_xml(annotation_file, xpath='.//object')[['name']]
    pos_df = pd.read_xml(annotation_file, xpath='.//bndbox')
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
    name: 物体名称
    x:    中心横坐标
    y:    中心纵坐标
    w:    宽度
    h:    高度
    """

    filenames = os.listdir(os.path.join(DATA_PATH, 'Annotations'))
    pbar = tqdm(filenames, total=len(filenames))
    for file in pbar:
        convert_annotation(file)

def padding_resize_img(img_path, img_size):
    """
    填充并修改尺寸
    """

    img_name = img_path.split('.')[0]
    img = cv2.imread(os.path.join(DATA_PATH, 'JPEGImages', img_path))
    h, w = img.shape[:2]
    padw, padh = 0, 0
    if h > w:
        padw = (h-w) // 2
        img = np.pad(img,((0,0),(padw,padw),(0,0)),'constant',constant_values=0)
    elif w > h:
        padh = (w - h) // 2
        img = np.pad(img,((padh,padh),(0,0),(0,0)), 'constant', constant_values=0)

    df = pd.read_csv(os.path.join(OUTPUT_LABEL_PATH, f'{img_name}.csv'))    
    if padw != 0:
        df['x'] = (df['x'] * w + padw) / h
        df['w'] = (df['w'] * w) / h
    elif padh != 0:
        df['y'] = (df['y'] * h + padh) / w
        df['h'] = (df['h'] * h) / w
    df.to_csv(os.path.join(OUTPUT_LABEL_PATH, f'{img_name}.csv'), index=None)

    img = cv2.resize(img, (img_size, img_size))
    cv2.imwrite(os.path.join(OUTPUT_IMG_PATH, img_path), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

def generate_img():
    """
    修改所有图片并修改标签
    """

    filenames = os.listdir(os.path.join(DATA_PATH, 'JPEGImages'))
    pbar = tqdm(filenames, total=len(filenames))
    for file in pbar:
        padding_resize_img(file, 448)


if __name__ == '__main__':
    generate_label()
    generate_img()
