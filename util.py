#!/usr/bin/env python
# --*-- encoding: utf-8 --*--
# @Author: Koorye
# @Date: 2021-10-27
# @Desc: 工具类，包含模型加载、IOU计算、输出核解码、可视化等功能

import cv2
import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from dataset import VOCDataset

from yolo_v1 import YoloV1


np.random.seed(1234)

OUTPUT_MODEL_PATH = 'output/model'
IMG_PATH = 'data/img'
LABEL_PATH = 'data/label'

CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

# 生成颜色字典
color_dic = {}
for each in CLASSES:
    r, g, b = np.random.randint(0, 255, 3)
    r, g, b = int(r), int(g), int(b)
    color_dic[each] = (r, g, b)


def calculate_iou(bbox1, bbox2):
    """
    计算bbox1和bbox2两个bbox的iou
    : param bbox1: 第1个候选框 (x1,y1,x2,y2)
    : param bbox2: 第2个候选框 (x1,y1,x2,y2)
    return: IOU
    """

    intersect_bbox = [0., 0., 0., 0.]
    if bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]:
        pass
    else:
        # 计算交集的xmin,ymin,xmax,ymax
        intersect_bbox[0] = max(bbox1[0], bbox2[0])
        intersect_bbox[1] = max(bbox1[1], bbox2[1])
        intersect_bbox[2] = min(bbox1[2], bbox2[2])
        intersect_bbox[3] = min(bbox1[3], bbox2[3])

    # 计算第1、2个候选框的面积
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    # 计算交集的面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) \
        * (intersect_bbox[3] - intersect_bbox[1])

    # 计算IOU
    if area_intersect > 0:
        return area_intersect / (area1 + area2 - area_intersect)
    else:
        return 0


def NMS(bboxes, scores, threshold=0.5):
    '''
    非极大值抑制
    : param bboxes: 预测框集 [n,4]，每一行为一个预测框 [x1,y1,x2,y2]
    : param scores: 分数，即预测概率 置信度x条件概率 [n,] -> [p1,p2,...]
    : param threshold: IOU阈值
    : return keep 张量，每个元素表示保留的预测框下标(所在的行数) -> [a1,a2,...]
    '''

    # 获取每个预测框的坐标并计算面积
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2-x1) * (y2-y1)

    # 对置信度倒序排序
    _, order = scores.sort(0, descending=True)
    keep = []

    # 从第一名的预测框开始
    while order.numel() > 0:
        # 将其加入列表选中
        if list(order.size()) == []:
            i = order
        else:
            i = order[0]
        keep.append(i)

        # 如果仅剩一个待选中的预测框，结束循环
        if order.numel() == 1:
            break

        # 将其余预测框的坐标归一到选中预测框的坐标范围内
        # 即其余预测框左上角坐标(x1,y1)>=(x1_0,y1_0)，(x2,y2)<=(x2_0,y2_0)
        # 其中x1,y1,x2,y2表示预测框的左上角和右下角的坐标
        # xi_0,yi_0表示选中的预测框的坐标
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        # 计算交集的面积
        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        # 计算IOU = 交集的面积 / (预测框1的面积+预测框2的面积-交集的面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到IOU小于阈值的预测框在排名中的在ovr列表中的下标
        ids = (ovr <= threshold).nonzero().squeeze()
        # 没有找到，结束循环
        if ids.numel() == 0:
            break
        # 更新待选预测框列表，筛选小于阈值的预测框(去除大于等于阈值的预测框，即与选中框重复的预测框)，同时删除选中框
        # 之所以+1，是为了使ovr的元素与order的元素对齐(ovr少了选中框位于下标0)
        order = order[ids+1]
    return torch.LongTensor(keep)


def label2bbox(pred):
    '''
    输出核转预测框
    : param pred: 预测框 [7,7,30] -> [x,y,w,h,conf,x,y,w,h,conf,p1,p2,...]
    : return: bbox [n,6] -> [id,x1,y1,x2,y2,conf]
    '''

    device = pred.device
    grid_num = 7
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1./grid_num
    pred = pred.data

    # 第一个预测框和第二个预测框的置信度 [7,7,30] -> [7,7] -> [7,7,1], [7,7,1]+[7,7,1] -> [7,7,2]
    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)

    # 置信度大于阈值的预测框置1
    mask1 = contain > 0.1
    # 所有预测框中置信度最大的置1
    mask2 = (contain == contain.max())
    # 置信度大于阈值或置信度最大(至少有一个预测框？)的预测框置1
    mask = (mask1+mask2).gt(0)

    # 遍历每个网格的每个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # 如果满足上述条件，即大于阈值或置信度最大
                if mask[i, j, b] == 1:
                    # 得到预测框 -> [x,y,w,h]
                    box = pred[i, j, b*5:b*5+4]
                    contain_prob = torch.FloatTensor(
                        [pred[i, j, b*5+4]]).to(device)
                    # 得到预测框位于网格的左上角坐标 -> 网格行、列数x每个网格的尺寸
                    xy = torch.FloatTensor([j, i]).to(device)*cell_size
                    box_xy = torch.FloatTensor(box.size()).to(device)
                    # 将x,y转换为绝对坐标
                    box[:2] = box[:2]*cell_size + xy
                    # 计算左上角和右下角的绝对坐标(x1,y1),(x2,y2)放入box_xy
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]

                    # 得到该网格预测物体的最大条件概率和对应物体类别的下标，均为标量
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    # 如果置信度x条件概率>0.1，将其作为一个预测框
                    if float((contain_prob*max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexs.append(torch.Tensor([cls_index]).to(device))
                        probs.append(contain_prob*max_prob)

    # 如果没有筛选出预测框，输出空
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
        return torch.cat([boxes, cls_indexs.unsqueeze(1), probs.unsqueeze(1)], -1)
    # 否则对预测框作拼接后经NMS筛选并返回
    # boxes [m,4] -> [x1,y1,x2,y1] 每个元素为第i个预测框的坐标信息
    # probs [m,] -> [p1,p2,...] 每个元素为第i个预测框的预测概率
    # cls_indexs [m,] -> [class1,class2,...] 每个元素为第i个预测框的类别下标
    # keep [n,] -> [a1,a2,...] 每个元素为筛选后保留的预测框的下标
    else:
        boxes = torch.cat(boxes, 0)
        probs = torch.cat(probs, 0)
        cls_indexs = torch.cat(cls_indexs, 0)
        keep = NMS(boxes, probs)
        return torch.cat([cls_indexs[keep].unsqueeze(1), boxes[keep], probs[keep].unsqueeze(1)], -1)


def load_model(historical_epoch, device):
    """
    加载模型
    : param historical_epoch: 历史训练次数，0表示不加载历史模型，>0表示对应训练次数的模型，-1表示最近一次训练的模型
    : param device: 设备
    : return: yolo, last_epoch 加载的模型和对应的训练次数
    """

    yolo = YoloV1().to(device)
    if historical_epoch == 0:
        last_epoch = 0
        return yolo, last_epoch

    if historical_epoch > 0:
        last_epoch = historical_epoch
    elif historical_epoch == -1:
        epoch_files = os.listdir(OUTPUT_MODEL_PATH)
        last_epoch = 0
        for file in epoch_files:
            file = file.split('.')[0]
            if file.startswith('epoch'):
                epoch = int(file[5:])
                if epoch > last_epoch:
                    last_epoch = epoch

    yolo.load_state_dict(torch.load(os.path.join(
        OUTPUT_MODEL_PATH, f'epoch{last_epoch}.pth')))
    return yolo, last_epoch


def draw_box(img, pos1, pos2, text, conf, color, width):
    """
    绘制有标签、颜色的矩形框
    : param img: 要绘制的图片
    : param pos1: 左上角坐标 (x1,y1)
    : param pos2: 右下角坐标 (x2,y2)
    : param text: 注释文字
    : param conf: 概率
    : param color: 颜色
    : param width: 宽度
    : return: img 绘制后的图片
    """

    text_len = len(text)
    img = cv2.rectangle(img, pos1, pos2, color, width)
    img = cv2.rectangle(img, (int(
        pos1[0]-width/2), pos1[1]-15), (int(pos1[0]+text_len*5.5+24), pos1[1]), color, -1)
    img = cv2.putText(
        img, text, (pos1[0], pos1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
    img = cv2.putText(img, '{:.2f}'.format(conf), (int(
        pos1[0]+text_len*5.5), pos1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
    return img


def draw_bbox(img, bbox):
    """
    生成图片和bbox
    : param img: 要绘制的图片 [448,448,3]
    : param bbox: 候选框 [n, 6] -> [id,x1,y1,x2,y2,p]
    : return: img 绘制后的图片
    """

    height, width = img.shape[:2]
    bbox_df = pd.DataFrame(bbox, columns=['id', 'x1', 'y1', 'x2', 'y2', 'p'])

    for _, row in bbox_df.iterrows():
        name = CLASSES[int(row['id'])]
        x1, y1 = int(row['x1'] * width), int(row['y1'] * height)
        x2, y2 = int(row['x2'] * width), int(row['y2'] * height)
        p = int(row['p'])
        img = draw_box(img, (x1, y1), (x2, y2), name,
                       float(row['p']), color_dic[name], p*5)
    return img


def calculate_acc(output, target, p=.1, iou_shresh=.5):
    """
    计算预测值和实际值之间的准确率和召回率
    : param output: 预测值 [7,7,30]
    : param target: 实际值 [7,7,30]
    : param p: 置信度阈值，置信度>=p的框被认为是正样本即预测出的样本
    : param iou_shresh，IOU阈值，当预测框和实际框的阈值>=IOU则认为预测框预测正确
    : return tp,m,n 预测正确的样本数、识别出的样本数、实际的样本数，返回用于后续的累加计算
    """

    output_bbox = label2bbox(output).detach().numpy()
    target_bbox = label2bbox(target).detach().numpy()

    output_df = pd.DataFrame(output_bbox, columns=[
                             'id', 'x1', 'y1', 'x2', 'y2', 'p']).astype(float)
    target_df = pd.DataFrame(target_bbox, columns=[
                             'id', 'x1', 'y1', 'x2', 'y2', 'p']).astype(float)
    output_df = output_df[output_df['p'] >= p]
    target_df = target_df[target_df['p'] >= p]

    # 识别出的标签数量，percision = tp/(tp+fp) = tp/m
    # precision 精度即识别出的标签中识别正确的比例
    m = len(output_df)
    # 标签的数量，recall = tp/(tp+fn) = tp/n
    # recall 召回(查全)率即所有标签中被识别出的标签的比例
    n = len(target_df)

    # 预测正确的样本数，即实际标签中被预测到的标签数
    tp = 0
    # 遍历所有预测框和实际框的两两组合
    for _, target_row in target_df.iterrows():
        target_id = int(target_row['id'])
        output_df_ = output_df[output_df['id'] == target_id]
        for _, output_row in output_df_.iterrows():
            # 所有两个框所属类别相同
            if int(output_row['id']) == int(target_row['id']):
                bbox1 = (float(output_row['x1']), float(output_row['y1']),
                         float(output_row['x2']), float(output_row['y2']))
                bbox2 = (float(target_row['x1']), float(target_row['y1']),
                         float(target_row['x2']), float(target_row['y2']))
                iou = calculate_iou(bbox1, bbox2)
                if iou >= iou_shresh:
                    tp += 1
                    break
    return tp, m, n


def calculate_acc_from_batch(output, target, p=.1, iou_shresh=.5):
    """
    计算预测值和实际值之间的准确率和召回率
    : param output: 预测值 [b,7,7,30]
    : param target: 实际值 [b,7,7,30]
    : param p: 置信度阈值，置信度>=p的框被认为是正样本即预测出的样本
    : param iou_shresh，IOU阈值，当预测框和实际框的阈值>=IOU则认为预测框预测正确
    : return tp,m,n 预测正确的样本数、识别出的样本数、实际的样本数，返回用于后续的累加计算
    """

    with torch.no_grad():
        b_size = output.size(0)
        tp, m, n = 0, 0, 0
        for b in range(b_size):
            tp_, m_, n_ = calculate_acc(
                output[b], target[b], p=p, iou_shresh=iou_shresh)
            tp += tp_
            m += m_
            n += n_
        return tp, m, n


if __name__ == '__main__':
    device = torch.device('cuda')
    cpu = torch.device('cpu')
    dataset = VOCDataset(mode='test')
    dataloader = DataLoader(dataset, shuffle=False, batch_size=8)
    precisions, recalls = [], []
    with torch.no_grad():
        for epoch in range(1,43+1,5):
            yolo, _ = load_model(epoch, device)
            yolo.eval()

            pbar = tqdm(dataloader, total=len(dataloader))
            tp, m, n = 0,0,0
            for index, (data, target) in enumerate(pbar):
                output = yolo(data.to(device))
                tp_,m_,n_ = calculate_acc_from_batch(output.to(cpu), target.to(cpu))
                tp += tp_
                m += m_
                n += n_
            precisions.append(tp/m)
            recalls.append(tp/n)
    plt.plot(precisions, label='precision')
    plt.plot(recalls, label='recall')
    plt.legend()
    plt.show()
