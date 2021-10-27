import os
import torch

from util import labels2bbox, show
from yolo_v1 import YoloV1
from dataset import VOCDataset

OUTPUT_MODEL_PATH = 'output/model'

test_data = VOCDataset('test')
data, _ = VOCDataset.__getitem__(123)
yolo = YoloV1()

epoch_files = os.listdir(OUTPUT_MODEL_PATH)
last_epoch = 0
for file in epoch_files:
    if file.startswith('epoch'):
        epoch = int(file[5:])
        if epoch > last_epoch:
            last_epoch = epoch

yolo.load_state_dict(torch.load(os.path.join(OUTPUT_MODEL_PATH, f'epoch{last_epoch}.pth')))
output = yolo(data.unsqueeze(0))

bbox = labels2bbox(output.squeeze(0))
show(data, bbox)