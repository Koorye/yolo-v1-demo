#!/usr/bin/env python
# --*-- encoding: utf-8 --*--
# @Author: Koorye
# @Date: 2021-10-27
# @Desc: 支持矢量化计算的Yolo V1 loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """
    并行计算多个候选框间的IOU
    : param box1: 第一组候选框 [N,4]
    : param box2: 第二组候选框 [M,4]
    : return: IOU [N,M]，其中第n行第j列的元素表示box1[n]和box2[m]的IOU
    """

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        # [N,2] -> [N,1,2] -> [N,M,2]
        box1[:, :2].unsqueeze(1).expand(N, M, 2),
        # [M,2] -> [1,M,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),
    )
    rb = torch.min(
        # [N,2] -> [N,1,2] -> [N,M,2]
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        # [M,2] -> [1,M,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )
    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloV1Loss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloV1Loss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def forward(self, predict, target):
        """
        :param predict: 预测的输出核 [b,7,7,30]
        :param target: 标签的输出核 [b,7,7,30]
        :return: loss
        """

        device = predict.device
        N = predict.size(0)

        # 将target中置信度>0的候选框对应的网格位置置为True，即保留有物体的网格
        coord_mask = target[:, :, :, 4] > 0
        # 将target中置信度=0的候选框对应的网格位置置为True，即保留无物体的网格
        noobj_mask = target[:, :, :, 4] == 0

        # 将mask扩展至最后一维 [b,S,S] -> [b,S,S,30]
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)

        # 将predict经coord_mask去除没有物体的网格，筛选出有物体的网格
        # [b,S,S,30] -> [b*S*S,30] 将输出核最后一维的每一列抽出，组成二维矩阵
        # 其中矩阵的每一行代表一个网格，每一列代表各个特征[x1,y1,w1,h1,p1,x2,y2,w2,h2,p2,p...]
        # 之后再将其最后一维的前10个元素抽出
        # [b*S*S,10] -> [b*S*S*2,5] 组成二维矩阵
        # 其中矩阵的每一行代表一个候选框，每一列代表候选框的各个特征[x,y,w,h,p]
        coord_pred = predict[coord_mask].view(-1, 30)
        coord_box_pred = coord_pred[:, :10].contiguous().view(-1, 5)
        coord_class_pred = coord_pred[:, 10:]

        # target同上
        coord_target = target[coord_mask].view(-1, 30)
        coord_box_target = coord_target[:, :10].contiguous().view(-1, 5)
        coord_class_target = coord_target[:, 10:]

        # 筛选出没有物体的网格
        # [b,S,S,30] -> [b*S*S,30] 将输出核最后一维的每一列抽出，组成二维矩阵
        # 其中矩阵的每一行代表一个网格，每一列代表各个特征[x1,y1,w1,h1,p1,x2,y2,w2,h2,p2,p...]
        # target同上
        # noobj_pred = predict.view(noobj_mask.size())[noobj_mask].view(-1, 30)
        noobj_pred = predict[noobj_mask].view(-1, 30)
        noobj_target = target[noobj_mask].view(-1, 30)

        # 设置尺寸同样为[b*S*S,30]的BoolTensor
        # 将第4、9列，即置信度对应的列置1，其余列置0
        noobj_pred_mask = torch.BoolTensor(noobj_pred.size()).to(device)
        noobj_pred_mask.zero_()
        noobj_pred_mask[:, 4] = 1
        noobj_pred_mask[:, 9] = 1

        # 筛选出置信度对应的列，计算MSE loss
        noobj_pred_c = noobj_pred[noobj_pred_mask]
        noobj_target_c = noobj_target[noobj_pred_mask]
        noobj_loss = F.mse_loss(noobj_pred_c, noobj_target_c, reduction='sum')

        # 设置尺寸为[b*S*S*2,5]的BoolTensor
        # 分别用于探测物体的候选框和不用于探测物体的候选框的计算
        coord_response_mask = torch.BoolTensor(coord_box_target.size()).to(device)
        coord_response_mask.zero_()
        coord_not_response_mask = torch.BoolTensor(coord_box_target.size()).to(device)
        coord_not_response_mask.zero_()

        # 设置尺寸为[b*S*S*2,5]的BoolTensor
        # 用于IOU计算结果的存储
        box_target_iou = torch.zeros(coord_box_target.size()).to(device)

        for i in range(0, coord_box_target.size(0), 2):
            # 每次取出predict中属于同一个网格的两个候选框
            box1 = coord_box_pred[i:i + 2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            # 计算其左上角坐标
            box1_xyxy[:, :2] = box1[:, :2] - 0.5 * box1[:, 2:4]
            # 计算其右下角坐标
            box1_xyxy[:, 2:4] = box1[:, :2] + 0.5 * box1[:, 2:4]

            # 每次取出target的一个候选框
            box2 = coord_box_target[i].view(-1, 5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            # 计算其左上角坐标
            box2_xyxy[:, :2] = box2[:, :2] - 0.5 * box2[:, 2:4]
            # 计算其右下角坐标
            box2_xyxy[:, 2:4] = box2[:, :2] + 0.5 * box2[:, 2:4]

            # 计算IOU，由于候选框属于同一个网络，相对坐标相同，故不需要转换为绝对坐标
            # 返回值为尺寸为[2,1]的矩阵
            # [[box11] * [box2] -> [[iou(box11,box2)]
            #  [box12]]             [iou(box12,box2)]]
            # 分别表示predict中第1、2个候选框和target中第1个候选框的IOU
            iou = compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])

            # 对第0维取最大值，即取得IOU最大的候选框和对应下标
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.to(device)

            # 将用于探测物体的候选框对应位置1，其余位为0
            coord_response_mask[i + max_index] = 1
            # 将不用于探测物体的候选框对应位置1，其余位为0
            coord_not_response_mask[i + 1 - max_index] = 1

            # 存储IOU最大值的计算结果到候选框对应位置
            box_target_iou[i + max_index, torch.LongTensor([4]).to(device)] = (max_iou).data.to(device)
        box_target_iou = Variable(box_target_iou).to(device)

        # 筛选出用于探测物体的候选框
        coord_box_pred_response = coord_box_pred[coord_response_mask].view(
            -1, 5)

        # 筛选出predict中用于探测物体的候选框对应的IOU
        box_target_response_iou = box_target_iou[coord_response_mask].view(
            -1, 5)

        # 筛选出target中用于探测物体的候选框
        box_target_response = coord_box_target[coord_response_mask].view(-1, 5)

        # 计算用于探测物体的候选框的置信度与IOU之间的MSE loss
        contain_loss = F.mse_loss(
            coord_box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum')

        # 计算坐标中心点的MSE loss和宽高开平方后的MSE loss
        loc_loss = F.mse_loss(coord_box_pred_response[:, :2], box_target_response[:, :2], reduction='sum') + \
            F.mse_loss(torch.sqrt(coord_box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]),
                       reduction='sum')

        # 同样筛选出不用于探测物体的候选框
        coord_box_pred_not_response = coord_box_pred[coord_not_response_mask].view(
            -1, 5)
        box_target_not_response = coord_box_target[coord_not_response_mask].view(
            -1, 5)

        # 将target的置信度置0 (表示没有物体)
        box_target_not_response[:, 4] = 0

        # 计算置信度的MSE loss
        not_contain_loss = F.mse_loss(coord_box_pred_not_response[:, 4], box_target_not_response[:, 4],
                                      reduction='sum')

        # 计算分类的条件概率MSE loss
        class_loss = F.mse_loss(
            coord_class_pred, coord_class_target, reduction='sum')

        # loss加权求和
        loss = (self.l_coord * loc_loss + 2 * contain_loss + not_contain_loss
                + self.l_noobj * noobj_loss + class_loss) / N

        return loss



if __name__ == '__main__':
    from yolo_v1 import YoloV1
    from dataset import VOCDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = YoloV1Loss(7, 2, 5, .5).to(device)
    
    dataset = VOCDataset()
    data, target = dataset.__getitem__(0)

    yolo = YoloV1().to(device)
    output = yolo(data.unsqueeze(0).to(device))

    # print(target[:,:,0])
    # print(target[:,:,1])
    # print(target[:,:,2])
    # print(target[:,:,3])
    # print(target[:,:,10:].sum(-1))
    # print(output)

    loss = loss(output.to(device), target.unsqueeze(0).to(device))
    print(loss)
