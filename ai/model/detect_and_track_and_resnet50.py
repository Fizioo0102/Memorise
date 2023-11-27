import torch
print(torch.cuda.is_available())

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from rembg import remove

# 모델 로딩
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x')
model_resnet = models.wide_resnet50_2(pretrained=True).eval().cuda()  # ResNet-50으로 특성 추출

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 로컬 이미지 특성 추출
# local_image = cv2.imread('C:/Users/SSAFY/Desktop/yolotest/imagetest2.jpg')  # 이미지 경로 입력
local_image = cv2.imread('C:/Users/SSAFY/Desktop/yolotest/cup1.jpg')  # 이미지 경로 입력
local_image = remove(local_image)

local_image_rgb = cv2.cvtColor(local_image, cv2.COLOR_BGR2RGB)
tensor_image = transform(local_image_rgb).unsqueeze(0).cuda()
features_local = model_resnet(tensor_image)

# 로컬 이미지에서 객체 탐지
results_local = model_yolo(local_image_rgb)
if len(results_local.pred[0]) > 0:  # 객체가 탐지된 경우
    x1_local, y1_local, x2_local, y2_local = map(int, results_local.pred[0][0][:4])
    mask_local = np.zeros_like(local_image_rgb)
    mask_local[y1_local:y2_local, x1_local:x2_local] = 1  # 탐지된 객체 부분만 1로 설정
    detected_local_obj = local_image_rgb[y1_local:y2_local, x1_local:x2_local]  # Detected object from local image
    cv2.imwrite("detected_object.jpg", cv2.cvtColor(detected_local_obj, cv2.COLOR_RGB2BGR))  # Save the detected object
    local_image_rgb = local_image_rgb * mask_local

tensor_image = transform(local_image_rgb).unsqueeze(0).cuda()
features_local = model_resnet(tensor_image)

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

        # 여기서 roi_rgb를 정의합니다.
        roi = frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        mask_roi = np.zeros_like(frame)
        mask_roi[y1:y2, x1:x2] = 1  # 탐지된 객체 부분만 1로 설정
        roi_rgb_masked = roi_rgb * mask_roi[y1:y2, x1:x2]

        tensor_roi = transform(roi_rgb_masked).unsqueeze(0).cuda()
        features_roi = model_resnet(tensor_roi)
        
        similarity = get_similarity(features_local, features_roi)
        label = f"Sim: {similarity.item():.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    cv2.imshow('YOLOv5 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()