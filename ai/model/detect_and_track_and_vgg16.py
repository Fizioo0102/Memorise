import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from rembg import remove



# from PIL import Image

# input = Image.open('testImg.jpg') # load image
# output = remove(input) # remove background
# output.save('rembg.PNG') # save image

# 모델 로딩
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x')
model_vgg = models.vgg16(pretrained=True).features.eval().cuda()  # VGG16으로 특성 추출

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# # 로컬 이미지 특성 추출
local_image = cv2.imread('C:/Users/SSAFY/Desktop/yolotest/cup.PNG')  # 이미지 경로 입력
local_image = remove(local_image)
local_image_rgb = cv2.cvtColor(local_image, cv2.COLOR_BGR2RGB)
tensor_image = transform(local_image_rgb).unsqueeze(0).cuda()
features_local = model_vgg(tensor_image)

def get_similarity(features1, features2):
    cos_sim = nn.functional.cosine_similarity(features1, features2, dim=1)
    return cos_sim.mean()


# 웹캠에서 비디오 가져오기
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # 너비 설정
cap.set(4, 1080)  # 높이 설정

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO로 객체 탐지
    results = model_yolo(frame)

    for *xyxy, conf, cls in results.pred[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        roi = frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        tensor_roi = transform(roi_rgb).unsqueeze(0).cuda()
        features_roi = model_vgg(tensor_roi)
        
        similarity = get_similarity(features_local, features_roi)
        label = f"Sim: {similarity.item():.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    cv2.imshow('YOLOv5 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()