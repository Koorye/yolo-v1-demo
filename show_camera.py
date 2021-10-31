import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from util import load_model, label2bbox, draw_bbox


vc = cv2.VideoCapture(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
yolo,_ = load_model(-1, device)

trans = transforms.Compose([
    transforms.ToPILImage(mode='RGB'),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

with torch.no_grad():
    yolo.eval()
    while True:
        ret, img = vc.read()
        h, w = img.shape[:2]
        padw, padh = 0, 0
        if h > w:
            padw = (h-w) // 2
            img = np.pad(img, ((0, 0), (padw, padw), (0, 0)),
                         'constant', constant_values=0)
        elif w > h:
            padh = (w - h) // 2
            img = np.pad(img, ((padh, padh), (0, 0), (0, 0)),
                         'constant', constant_values=0)

        img = trans(img).to(device)
        output = yolo(img.unsqueeze(0))
        output_bbox = label2bbox(output.to(cpu).squeeze(0))

        img = np.uint8(img.transpose(0, 1).transpose(1, 2).cpu().numpy() * 255)
        output_img = draw_bbox(img, output_bbox)
        cv2.imshow('img', output_img)

        c = cv2.waitKey(1)
        if c == 27:
            break

vc.release()
cv2.destroyAllWindows()
