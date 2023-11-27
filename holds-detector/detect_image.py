import cv2

import os
import sys
import numpy as np
from model import create_model

import torch
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

if __name__ == '__main__':
    try:
        image_path = sys.argv[1]
        image_name = os.path.basename(image_path)
    except:
        print('Please enter a file path')


model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

image = cv2.imread(image_path)
orig_image = image.copy()
# BGR to RGB
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
# make the pixel range between 0 and 1
image /= 255.0
# bring color channels to front
image = np.transpose(image, (2, 0, 1)).astype(np.float32)
# convert to tensor
image = torch.tensor(image, dtype=torch.float).cuda()
# add batch dimension
image = torch.unsqueeze(image, 0)
with torch.no_grad():
    outputs = model(image.to(DEVICE))
# load all detection to CPU for further operations
outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
# carry further only if there are detected boxes
detection_threshold = 0.8

if len(outputs[0]['boxes']) != 0:
    boxes = outputs[0]['boxes'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()

    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    draw_boxes = boxes.copy()

    pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
    

    for j, box in enumerate(draw_boxes):
        class_name = pred_classes[j]
        color = int('920310', 16)

        cv2.rectangle(orig_image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color, 2)
    cv2.imwrite(f"detect_output/{image_name}.jpg", orig_image)
    cv2.imshow('Prediction', orig_image)
    cv2.waitKey(0)
else:
    print("nothing detected")