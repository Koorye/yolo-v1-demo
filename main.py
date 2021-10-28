#!/usr/bin/env python
# --*-- encoding: utf-8 --*--
# @Author: Koorye
# @Date: 2021-10-27
# @Desc: Yolo V1的训练、测试、可视化和保存

import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import visdom

from dataset import VOCDataset
from show_yolo import draw_img_with_bbox
from yolo_v1_loss_vectorization import YoloV1Loss
from util import load_model

# 初始化参数
# EPOCHS: 总的训练次数
# HISTORICAL_EPOCHS: 历史训练次数，用于模型的加载
# - -1表示最近一次训练的模型
# - 0表示不加载历史模型
# - >0表示对应训练次数的模型
# SAVE_EVERY: 保存频率，每训练多少次保存一次
# BATCH_SIZE: 每次喂入的数据量
# LR: 学习率
EPOCHS = 200
HISTORICAL_EPOCHS = -1
SAVE_EVERY = 1
BATCH_SIZE = 7
LR = 1e-3

# OUTPUT_MODEL_PATH: 输出的模型路径
# CLASSES: 类别列表
OUTPUT_MODEL_PATH = 'output/model'
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

print('============================================')
if not os.path.exists(OUTPUT_MODEL_PATH):
    os.makedirs(OUTPUT_MODEL_PATH)

# 检测设备
if torch.cuda.is_available():
    print('CUDA已启用')
    device = torch.device('cuda')
else:
    print('CUDA不可用，使用CPU')
    device = torch.device('cpu')

# 加载数据集和加载器
print('加载数据集...')
train_data = VOCDataset('train')
test_data = VOCDataset('test')
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# 加载模型
print('加载模型和优化器...')
yolo, last_epoch = load_model(HISTORICAL_EPOCHS, device)
criterion = YoloV1Loss(7, 2, 5, 0.5, device)
optim = torch.optim.SGD(yolo.parameters(), lr=LR,
                        momentum=.9, weight_decay=5e-4)

# 加载调度器
def lr_lambda(ep):
    current_epoch = ep + last_epoch
    # 预热期学习率由 1e-4 -> 1e-3
    if current_epoch == 0:
        return .1
    elif current_epoch == 1:
        return .5
    elif current_epoch == 2:
        return .75
    # 之后学习率阶梯下降 1e-3 -> 1e-4 -> 1e-5
    elif current_epoch < 30:
        return 1.
    elif current_epoch < 40:
        return .1
    else:
        return .01
    

yolo_lr = LambdaLR(optim, lr_lambda=lr_lambda)

print('开启可视化...')
viz = visdom.Visdom()

# 开始训练
print('============================================')
train_loss, test_loss = [], []
for epoch in range(last_epoch+1, EPOCHS+last_epoch+1):

    # 训练
    yolo.train()
    pbar = tqdm(enumerate(train_loader), total=len(
        train_loader), desc=f'第{epoch}次训练')
    for index, (data, label) in pbar:
        data = data.to(device)
        label = label.float().to(device)

        output = yolo(data)

        loss = criterion(output, label)
        if np.isnan(loss.item()):
            print('梯度爆炸！')
            exit(-1)
        train_loss.append(loss.item())
        if len(train_loss) > 1000:
            train_loss.pop(0)
        optim.zero_grad()
        loss.backward()
        optim.step()

        viz.line(train_loss, list(range(len(train_loss))), win='训练Loss', opts={'title': '训练Loss'})

    yolo_lr.step()

    # 测试
    with torch.no_grad():
        yolo.eval()

        total_loss = 0
        pbar = tqdm(enumerate(test_loader), total=len(
            test_loader), desc=f'第{epoch}次测试')
        for index, (data, label) in pbar:
            data = data.to(device)
            label = label.to(device)

            output = yolo(data)
            loss = criterion(output, label)
            total_loss += loss.item()

        total_loss /= len(test_loader)
        test_loss.append(total_loss)
        viz.line(test_loss, list(range(len(test_loss))), win='测试Loss', opts={'title': '测试Loss'})

        torch.cuda.empty_cache()

        # 可视化预测效果
        pred_imgs, target_imgs = draw_img_with_bbox(yolo, 8, f'epoch{epoch}', save=True)
        viz.images(pred_imgs, win='预测图片', opts={'title':'预测图片'})
        viz.images(target_imgs, win='实际图片', opts={'title':'实际图片'})
    
    # 保存模型
    if epoch % SAVE_EVERY == 0:
        torch.save(yolo.state_dict(), os.path.join(
            OUTPUT_MODEL_PATH, f'epoch{epoch}.pth'))
