import cv2
import numpy as np
import os
import pandas as pd
import torch


np.random.seed(1234)

IMG_PATH = 'data/img'
LABEL_PATH = 'data/label'

CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

color_dic = {}

for each in CLASSES:
    r, g, b = np.random.randint(0, 255, 3)
    r, g, b = int(r), int(g), int(b)
    color_dic[each] = (r,g,b)

def draw_box(img, pos1, pos2, text, color, width):
    """
    绘制有标签、颜色的矩形框
    """
    text_len = len(text)
    img = cv2.rectangle(img, pos1, pos2, color, width)
    img = cv2.rectangle(img, (int(pos1[0]-width/2),pos1[1]-15), (int(pos1[0]+text_len*5+2+width/2),pos1[1]), color, -1)
    img = cv2.putText(img, text, (pos1[0], pos1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
    return img

def calculate_iou(bbox1,bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的交集
    if bbox1[2]<bbox2[0] or bbox1[0]>bbox2[2] or bbox1[3]<bbox2[1] or bbox1[1]>bbox2[3]:
        pass
    else:
        intersect_bbox[0] = max(bbox1[0],bbox2[0])
        intersect_bbox[1] = max(bbox1[1],bbox2[1])
        intersect_bbox[2] = min(bbox1[2],bbox2[2])
        intersect_bbox[3] = min(bbox1[3],bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # input()

    if area_intersect>0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0

def NMS(bbox, conf_thresh=0.1, iou_thresh=0.3):
    """bbox数据格式是(n,25),前4个是(x1,y1,x2,y2)的坐标信息，第5个是置信度，后20个是类别概率
    :param conf_thresh: cls-specific confidence score的阈值
    :param iou_thresh: NMS算法中iou的阈值
    """
    n = bbox.size()[0]
    bbox_prob = bbox[:,5:].clone()  # 类别预测的条件概率
    bbox_confi = bbox[:, 4].clone().unsqueeze(1).expand_as(bbox_prob)  # 预测置信度
    bbox_cls_spec_conf = bbox_confi*bbox_prob  # 置信度*类别条件概率=cls-specific confidence score整合了是否有物体及是什么物体的两种信息
    bbox_cls_spec_conf[bbox_cls_spec_conf<=conf_thresh] = 0  # 将低于阈值的bbox忽略
    for c in range(20):
        rank = torch.sort(bbox_cls_spec_conf[:,c],descending=True).indices
        for i in range(98):
            if bbox_cls_spec_conf[rank[i],c]!=0:
                for j in range(i+1,98):
                    if bbox_cls_spec_conf[rank[j],c]!=0:
                        iou = calculate_iou(bbox[rank[i],0:4],bbox[rank[j],0:4])
                        if iou > iou_thresh:  # 根据iou进行非极大值抑制抑制
                            bbox_cls_spec_conf[rank[j],c] = 0
    bbox = bbox[torch.max(bbox_cls_spec_conf,dim=1).values>0]  # 将20个类别中最大的cls-specific confidence score为0的bbox都排除
    bbox_cls_spec_conf = bbox_cls_spec_conf[torch.max(bbox_cls_spec_conf,dim=1).values>0]
    res = torch.ones((bbox.size()[0],6))
    res[:,1:5] = bbox[:,0:4]  # 储存最后的bbox坐标信息
    res[:,0] = torch.argmax(bbox[:,5:],dim=1).int()  # 储存bbox对应的类别信息
    res[:,5] = torch.max(bbox_cls_spec_conf,dim=1).values  # 储存bbox对应的class-specific confidence scores
    return res

def labels2bbox(matrix, use_nms=True):
    """
    将网络输出的7*7*30的数据转换为bbox的(98,25)的格式，然后再将NMS处理后的结果返回
    :param matrix: 注意，输入的数据中，bbox坐标的格式是(px,py,w,h)，需要转换为(x1,y1,x2,y2)的格式再输入NMS
    :return: 返回NMS处理后的结果
    """
    if matrix.size()[0:2]!=(7,7):
        raise ValueError("Error: Wrong labels size:",matrix.size())
    bbox = torch.zeros((98,25))
    # 先把7*7*30的数据转变为bbox的(98,25)的格式，其中，bbox信息格式从(px,py,w,h)转换为(x1,y1,x2,y2),方便计算iou
    for i in range(7):  # i是网格的行方向(y方向)
        for j in range(7):  # j是网格的列方向(x方向)
            bbox[2*(i*7+j),0:4] = torch.Tensor([(matrix[i, j, 0] + j) / 7 - matrix[i, j, 2] / 2,
                                                (matrix[i, j, 1] + i) / 7 - matrix[i, j, 3] / 2,
                                                (matrix[i, j, 0] + j) / 7 + matrix[i, j, 2] / 2,
                                                (matrix[i, j, 1] + i) / 7 + matrix[i, j, 3] / 2])
            bbox[2 * (i * 7 + j), 4] = matrix[i,j,4]
            bbox[2*(i*7+j),5:] = matrix[i,j,10:]
            bbox[2*(i*7+j)+1,0:4] = torch.Tensor([(matrix[i, j, 5] + j) / 7 - matrix[i, j, 7] / 2,
                                                (matrix[i, j, 6] + i) / 7 - matrix[i, j, 8] / 2,
                                                (matrix[i, j, 5] + j) / 7 + matrix[i, j, 7] / 2,
                                                (matrix[i, j, 6] + i) / 7 + matrix[i, j, 8] / 2])
            bbox[2 * (i * 7 + j)+1, 4] = matrix[i, j, 9]
            bbox[2*(i*7+j)+1,5:] = matrix[i,j,10:]
    if use_nms:
        return NMS(bbox)  # 对所有98个bbox执行NMS算法，清理cls-specific confidence score较低以及iou重合度过高的bbox
    
    res = torch.zeros(98, 6)
    res[:,0] = torch.argmax(bbox[:,5:],dim=1).int()  # 储存bbox对应的类别信息
    res[:,1:6] = bbox[:, :5]
    return res


def show(img, bbox):
    height, width = img.shape[:2]
    bbox_df = pd.DataFrame(bbox, columns=['id', 'x1', 'y1', 'x2', 'y2', 'p'])

    for _, row in bbox_df.iterrows():
        name = CLASSES[int(row['id'])]
        x1, y1 = int(row['x1'] * width), int(row['y1'] * height)
        x2, y2 = int(row['x2'] * width), int(row['y2'] * height)
        p = int(row['p'])
        img = draw_box(img, (x1,y1), (x2,y2), name, color_dic[name], p*5)
    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    from dataset import VOCDataset
    from yolo_v1 import YoloV1

    data, target = VOCDataset().__getitem__(0)
    yolo = YoloV1()
    output = yolo(data.unsqueeze(0))

    img = np.uint8(data.transpose(0,1).transpose(1,2).cpu().numpy() * 255)
    # bbox = labels2bbox(output.squeeze(0), use_nms=False)
    # bbox = labels2bbox(target, use_nms=False)
    bbox = labels2bbox(output.squeeze(0))

    show(img, bbox)
