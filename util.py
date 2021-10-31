#!/usr/bin/env python
# --*-- encoding: utf-8 --*--
# @Author: Koorye
# @Date: 2021-10-27
# @Desc: 工具类，包含模型加载、IOU计算、输出核解码、可视化等功能

import cv2
import os
import numpy as np
import pandas as pd
import torch

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


def NMS(bbox, conf_thresh=0.1, iou_thresh=0.3):
    """
    NMS非极大值抑制
    : param bbox: [n,25] -> [x1,y1,x2,y2,conf,p1,p2,...]
    : param conf_thresh: 置信度阈值，小于该阈值直接排除
    : param iou_thresh: IOU阈值，大于该阈值则排除置信度小的候选框
    : return: 抑制后的候选框 [m,6] -> [id,x1,y1,x2,y2,conf]
    """

    # 得到每个候选框的条件概率 [n,25] -> [n,20]
    bbox_prob = bbox[:, 5:].clone()

    # 得到每个候选框的置信度并扩展 [n,25] -> [n,] -> [n,1] -> [n,20]
    bbox_confi = bbox[:, 4].clone().unsqueeze(1).expand_as(bbox_prob)

    # 相乘得到每个候选框预测每个物体的概率 [n,20] * [n,20] -> [n,20]
    bbox_cls_spec_conf = bbox_confi*bbox_prob

    # 排除低于阈值的候选框 [n,20]
    bbox_cls_spec_conf[bbox_cls_spec_conf <= conf_thresh] = 0

    # 遍历每一种物体
    for c in range(20):
        # 对所有候选框进行排序，排序依据为该类物体的概率，倒序排序并取索引 [20] -> [r1,r2,...,r20]分别表示第i个候选框的排名
        rank = torch.sort(bbox_cls_spec_conf[:, c], descending=True).indices
        # 遍历每个候选框
        for i in range(98):
            # 如果第i个候选框的预测概率不为0
            if bbox_cls_spec_conf[rank[i], c] != 0:
                # 遍历小于该候选框排名的所有候选框
                for j in range(i+1, 98):
                    # 如果预测概率不为0
                    if bbox_cls_spec_conf[rank[j], c] != 0:
                        # 如果IOU大于阈值，将后者预测概率置0
                        iou = calculate_iou(
                            bbox[rank[i], 0:4], bbox[rank[j], 0:4])
                        if iou > iou_thresh:  # 根据iou进行非极大值抑制抑制
                            bbox_cls_spec_conf[rank[j], c] = 0

    # 筛选最大预测概率大于0(NMS抑制后有概率即表明有探测到物体)的候选框 [n,25] -> [m,25]
    # m为筛选后的候选框数量，即每一行是一个候选框
    bbox = bbox[torch.max(bbox_cls_spec_conf, dim=1).values > 0]

    # 筛选最大预测概率大于0的候选框 [n,25] -> [m,25]
    bbox_cls_spec_conf = bbox_cls_spec_conf[torch.max(
        bbox_cls_spec_conf, dim=1).values > 0]

    # 创建尺寸为[m,6]的矩阵用于存储筛选后的候选框
    res = torch.ones((bbox.size(0), 6))

    # 存储x1,y1,x2,y2
    res[:, 1:5] = bbox[:, 0:4]
    # 存储类别所属的id
    res[:, 0] = torch.argmax(bbox[:, 5:], dim=1).int()
    # 存储概率的最大值
    res[:, 5] = torch.max(bbox_cls_spec_conf, dim=1).values
    return res


def labels2bbox(matrix, use_nms=True):
    """
    将[7,7,30]的输出核转换为[m,6]的候选框
    : param matrix: 输出核 [7,7,30]
    : param use_nms: 是否使用NMS，不使用将保留所有候选框
    : return: 候选框 [m,6] -> [id,x1,y1,x2,y2,conf]，其中m表示候选框的数量
    """

    # 创建尺寸为[98,25]的矩阵用于存储候选框 -> [x1,y1,x2,y2,conf,p1,p2,...]
    bbox = torch.zeros((98, 25))
    for row in range(7):
        for col in range(7):
            # 将第i行第j列网格的坐标信息x,y(相对值),w,h
            # 转换为x1,y1,x2,y2
            # (x+col)/ncol - w/2 -> x1
            # (y+row)/nrow - h/2 -> y1
            # (x+col)/ncol + w/2 -> x2
            # (y+row)/nrow + h/2 -> y2
            # 存储到bbox对应位置
            bbox[2*(row*7+col), :4] = torch.Tensor([(matrix[row, col, 0]+col) / 7
                                                    - matrix[row, col, 2] / 2,
                                                    (matrix[row, col, 1]+row) / 7
                                                    - matrix[row, col, 3] / 2,
                                                    (matrix[row, col, 0]+col) / 7
                                                    + matrix[row, col, 2] / 2,
                                                    (matrix[row, col, 1]+row) / 7
                                                    + matrix[row, col, 3] / 2])

            # 存储置信度
            bbox[2*(row*7+col), 4] = matrix[row, col, 4]
            # 存储条件概率
            bbox[2*(row*7+col), 5:] = matrix[row, col, 10:]

            # 同理，存储第2个候选框到对应位置
            bbox[2*(row*7+col)+1, :4] = torch.Tensor([(matrix[row, col, 5]+col) / 7
                                                      - matrix[row,
                                                               col, 7] / 2,
                                                      (matrix[row,
                                                       col, 6]+row) / 7
                                                      - matrix[row,
                                                               col, 8] / 2,
                                                      (matrix[row,
                                                       col, 5]+col) / 7
                                                      + matrix[row,
                                                               col, 7] / 2,
                                                      (matrix[row,
                                                       col, 6]+row) / 7
                                                      + matrix[row, col, 8] / 2])
            # 存储置信度
            bbox[2*(row*7+col)+1, 4] = matrix[row, col, 9]
            # 存储条件概率
            bbox[2*(row*7+col)+1, 5:] = matrix[row, col, 10:]
    if use_nms:
        # NMS抑制 [98,25] -> [m,6]
        return NMS(bbox)

    # 不经过NMS，直接筛选结果
    res = torch.zeros(98, 6)
    # 获取预测概率最大的类别的id并存储
    res[:, 0] = torch.argmax(bbox[:, 5:], dim=1).int()
    # 获取每个类别的坐标和置信度x1,y1,x2,y2,conf并存储
    res[:, 1:6] = bbox[:, :5]
    return res


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
        yolo.load_state_dict(torch.load(os.path.join(
            OUTPUT_MODEL_PATH, f'epoch{historical_epoch}.pth')))
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

    output_bbox = labels2bbox(output)
    target_bbox = labels2bbox(target)
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
        for _, output_row in output_df.iterrows():
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
    from dataset import VOCDataset
    import datetime
    from torch.utils.data.dataloader import DataLoader

    dataset = VOCDataset('train')
    dataloader = DataLoader(dataset, shuffle=True, batch_size=8)
    data, target = next(iter(dataloader))

    device = torch.device('cpu')
    yolo, _ = load_model(0, device)
    output = yolo(data)

    start_time = datetime.datetime.now()
    print(calculate_acc_from_batch(output, target))
    end_time = datetime.datetime.now()
    print(end_time-start_time)

    start_time = datetime.datetime.now()
    print(calculate_acc_from_batch(target, target, iou_shresh=.5))
    end_time = datetime.datetime.now()
    print(end_time-start_time)
