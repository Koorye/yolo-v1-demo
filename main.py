import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import visdom

from dataset import VOCDataset
from yolo_v1 import YoloV1
from yolo_v1_loss_v3 import YoloV1Loss

EPOCHS = 100
HISTORICAL_EPOCHS = 0
SAVE_EVERY = 1
BATCH_SIZE = 1
LR = 1e-5
BETAS = (.5, .999)

OUTPUT_IMG_PATH = 'output/img'
OUTPUT_MODEL_PATH = 'output/model'
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

if not os.path.exists(OUTPUT_IMG_PATH):
    os.makedirs(OUTPUT_IMG_PATH)

if not os.path.exists(OUTPUT_MODEL_PATH):
    os.makedirs(OUTPUT_MODEL_PATH)

if torch.cuda.is_available():
    print('CUDA已启用')
    device = torch.device('cuda')
else:
    print('CUDA不可用，使用CPU')
    device = torch.device('cpu')

train_data = VOCDataset('train')
test_data = VOCDataset('test')

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

yolo = YoloV1().to(device)
criterion = YoloV1Loss(7, 2, 5, 0.5, device)
# optim = torch.optim.Adam(yolo.parameters(), lr=LR, betas=BETAS)
optim = torch.optim.SGD(yolo.parameters(), lr=LR, momentum=.9, weight_decay=5e-4)

    
if HISTORICAL_EPOCHS > 0:
    yolo.load_state_dict(torch.load(os.path.join(OUTPUT_MODEL_PATH, f'epoch{HISTORICAL_EPOCHS}.pth')))
elif HISTORICAL_EPOCHS == -1:
    epoch_files = os.listdir(OUTPUT_MODEL_PATH)
    last_epoch = 0
    for file in epoch_files:
        if file.startswith('epoch'):
            epoch = int(file[5:])
            if epoch > last_epoch:
                last_epoch = epoch
	yolo.load_state_dict(torch.load(os.path.join(OUTPUT_MODEL_PATH, f'epoch{last_epoch}.pth')))
    
viz = visdom.Visdom()

train_loss, test_loss = [], []
for epoch in range(1,EPOCHS+1):
    yolo.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'第{epoch}次训练')
    for index, (data, label) in pbar:
        data = data.to(device)
        label = label.float().to(device)
        print(data.size())
        print(label.size())

        output = yolo(data)

        loss = criterion(output, label)
        train_loss.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

        viz.line(train_loss, win='训练Loss', opts={'title':'训练Loss'})

    with torch.no_grad():
        yolo.eval()

        total_loss = 0
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f'第{epoch}次测试')
        for index, (data, label) in pbar:
            data = data.to(device)
            label = label.to(device)

            output = yolo(data)
            loss = criterion(output, label)
            total_loss += loss.item()

        total_loss /= len(test_loader)
        test_loss.append(total_loss)
        viz.line(total_loss, win='测试Loss', opts={'title':'测试Loss'})
        output = yolo()
    
    if epoch % SAVE_EVERY == 0:
        torch.save(yolo.state_dict(), os.path.join(OUTPUT_MODEL_PATH, f'epoch{epoch}.pth'))
