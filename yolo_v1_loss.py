#!/usr/bin/env python
# --*-- encoding: utf-8 --*--
# @Author: Koorye
# @Date: 2021-10-27
# @Desc: 不支持矢量化计算的Yolo V1 loss

import torch
from torch import nn

from util import calculate_iou

class YoloV1Loss(nn.Module):
    """
    Yolo V1损失函数的计算
    """

    def __init__(self):
        super(YoloV1Loss, self).__init__()

    def forward(self, pred, label):
        """
        : param pred: [b,30,7,7] 网络输出
        : param label: [b,30,7,7] 样本标签
        """

        # b_size: 喂入的样本数
        # ncol, nrow: 网格划分的列、行的数量
        b_size = label.size(0)
        nrow, ncol = label.size()[-2:]

        # coord_loss: 含有目标的bbox坐标损失
        # obj_conf_loss: 含有目标的bbox置信度损失
        # noobj_conf_loss: 不含目标的网络置信度损失
        # class_loss: 含有目标的网络类别损失
        noobj_conf_loss = torch.Tensor([0.])
        coord_loss = torch.Tensor([0.])
        obj_conf_loss = torch.Tensor([0.])
        class_loss = torch.Tensor([0.])

        for b in range(b_size):
            for row in range(nrow):
                for col in range(ncol):
                    # 标签在第row行、第col列的网格的置信度为1 (标签中有探测的物体)
                    if label[b, 4, row, col] == 1:
                        # [b,x,row,col]中的x分别代表
                        # x = 0,1,2,3: bbox1中心点的x,y和w,h
                        # x = 4: bbox1的conf
                        # x = 5,6,7,8: bbox2中心点的x,y和w,h
                        # x = 9: bbox2的conf
                        # 其中x,y为相对于该网格宽高的位置，w,h为相对于整张图片的宽、高，范围在0~1之间

                        # 于是有(pred[b,0,row,col]+col)/nrow - pred[b,2,row,col]/2
                        # -> (x + col) / ncol - w / 2
                        # -> bbox中心点的绝对坐标x - bbox宽度的一半
                        # -> 左上角的x坐标
                        # 其余坐标点的计算过程类似，从而计算得到 bbox=(x1,y1,x2,y2)
                        # 第一个预测框
                        bbox1 = ((pred[b, 0, row, col]+col)/ncol
                                 - pred[b, 2, row, col]/2,
                                 (pred[b, 1, row, col]+row)/nrow
                                 - pred[b, 3, row, col]/2,
                                 (pred[b, 0, row, col]+col)/ncol
                                 + pred[b, 2, row, col]/2,
                                 (pred[b, 1, row, col]+row)/nrow
                                 + pred[b, 3, row, col]/2)

                        # 第二个预测框
                        bbox2 = ((pred[b, 5, row, col]+col)/ncol
                                 - pred[b, 7, row, col]/2,
                                 (pred[b, 6, row, col]+row)/nrow
                                 - pred[b, 8, row, col]/2,
                                 (pred[b, 5, row, col]+col)/ncol
                                 + pred[b, 7, row, col]/2,
                                 (pred[b, 6, row, col]+row)/nrow
                                 + pred[b, 8, row, col]/2)

                        # 标签的实际框
                        label_bbox = ((label[b, 0, row, col]+col)/ncol
                                      - label[b, 2, row, col]/2,
                                      (label[b, 1, row, col]+row)/nrow
                                      - label[b, 3, row, col]/2,
                                      (label[b, 0, row, col]+col)/ncol
                                      + label[b, 2, row, col]/2,
                                      (label[b, 1, row, col]+row)/nrow
                                      + label[b, 3, row, col]/2)

                        # 计算交并比IOU
                        iou1 = calculate_iou(bbox1, label_bbox)
                        iou2 = calculate_iou(bbox2, label_bbox)

                        # 选择交并比大的负责探测物体
                        if iou1 >= iou2:
                            # 计算中心点x,y的MSELoss
                            coord_loss += 5 * torch.sum((pred[b, 0:2, row, col]
                                                         - label[b, 0:2, row, col])**2)
                            # 计算w,h的平方根的MSELoss
                            coord_loss += torch.sum((pred[b, 2:4, row, col].sqrt()
                                                    - label[b, 2:4, row, col].sqrt())**2)

                            # 计算置信度的MSELoss
                            obj_conf_loss += torch.sum(
                                (pred[b, 4, row, col] - iou1)**2)

                            noobj_conf_loss += .5 * \
                                torch.sum((pred[b, 9, row, col] - iou2)**2)
                        else:
                            # 计算中心点x,y的MSELoss
                            coord_loss += 5 * torch.sum((pred[b, 5:7, row, col]
                                                         - label[b, 5:7, row, col])**2)
                            # 计算w,h的平方根的MSELoss
                            coord_loss += torch.sum((pred[b, 7:9, row, col].sqrt()
                                                     - label[b, 7:9, row, col].sqrt())**2)

                            # 计算置信度的MSELoss
                            obj_conf_loss += torch.sum(
                                (pred[b, 9, row, col] - iou2)**2)

                            noobj_conf_loss += .5 * \
                                torch.sum((pred[b, 4, row, col] - iou1)**2)

                        # 计算分类概率的MSELoss
                        class_loss += torch.sum((pred[b, 10:, row, col]
                                                 - label[b, 10:, row, col])**2)

                    # 没有探测的物体
                    else:
                        noobj_conf_loss += .5 * \
                            torch.sum(pred[b, [4, 9], row, col]) ** 2

        loss = coord_loss + obj_conf_loss + noobj_conf_loss + class_loss
        return loss / b_size


if __name__ == '__main__':
    loss = YoloV1Loss()
    x, y = torch.randn(2, 30, 7, 7), torch.randn(2, 30, 7, 7)
    print(loss(x, y))
