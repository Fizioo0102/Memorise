import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pymongo import MongoClient
import timm

# MongoDB에 연결
client = MongoClient('mongodb://localhost:27017/')
db = client['test']
collection = db['yolo8']

# Siamese Network 초기화
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_network = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
        self.fc = nn.Linear(1408, 512)  # EfficientNetB2의 출력 차원을 1408로 수정

    def forward_one(self, x):
        x = self.base_network.forward_features(x)  # EfficientNet에서는 forward_features 메서드 사용
    
        # Global Average Pooling 적용하여 1x1x1408 형태로 변경
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    
        x = x.view(x.size(0), -1)  # 텐서를 평탄화
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
    
# YOLO8 모델 로딩
CONFIDENCE_THRESHOLD = 0.6
model = YOLO('yolov8x-seg.pt')


model_siamese = SiameseNetwork().cuda()
# 여러분의 Siamese Network 가중치 파일을 여기서 로드할 수 있습니다.
if os.path.exists('siamese_yolo8_weights.pth'):
    model_siamese.load_state_dict(torch.load('siamese_yolo8_weights.pth'))

model_siamese.eval()  # 평가 모드로 설정

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_folder = 'C:\\Users\\SSAFY\\Desktop\\image\\leejunyong'
all_features = []

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    local_image = cv2.imread(image_path)
    local_image_rgb = cv2.cvtColor(local_image, cv2.COLOR_BGR2RGB)

    results = model.predict(local_image_rgb)

    print(list(enumerate(results)))

    for i, r in enumerate(results):
        masks_data = r.masks.data
        orig_img = r.orig_img


        # 객체별로 원시 마스크 텐서를 사용하여 특성 벡터 추출 및 저장
        for obj_idx, obj_mask in enumerate(masks_data):
            obj_mask = obj_mask.cpu().numpy()

            # 원시 마스크의 크기를 객체 이미지의 크기에 맞게 조절
            obj_mask = cv2.resize(obj_mask, (orig_img.shape[1], orig_img.shape[0]))

            # 원시 마스크에서 1인 부분 추출
            object_region = orig_img * obj_mask[:, :, np.newaxis]

            # 전처리 및 특성 추출
            object_region = cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB)

            # 이미지를 uint8로 변환
            object_region = (object_region * 255).astype(np.uint8)

            object_region = transform(object_region).unsqueeze(0).cuda()
            features = model_siamese.forward_one(object_region)

            print(features)


            numpy_array_features = features.cpu().detach().numpy()
            list_features = numpy_array_features.tolist()[0]  # 중첩 리스트 제거
            all_features.append(list_features)
            break

        print(f"Object {i}: Extracted and saved {len(masks_data)} features")
    
# 데이터베이스에 저장
db_entry = {
    "id": 1,
    "name": os.path.basename(image_folder),
    "user": "이준용",
    "vector": all_features
}

db.yolo8.insert_one(db_entry)


    




    
    








# results = model.predict(source="0", save=True, show=True, stream=True)

# # 이미지에서 객체의 원시 마스크 텐서가 1인 부분을 추출하고 특성 벡터를 저장
# for i, r in enumerate(results):
#     masks_data = r.masks.data
#     orig_img = r.orig_img

#     # 특성 벡터를 저장할 디렉토리 생성
#     output_dir = f"C:/Users/SSAFY/Desktop/segmentImage/object_{i}"
#     os.makedirs(output_dir, exist_ok=True)

#     # 객체별로 원시 마스크 텐서를 사용하여 특성 벡터 추출 및 저장
#     for obj_idx, obj_mask in enumerate(masks_data):
#         obj_mask = obj_mask.cpu().numpy()

#         # 원시 마스크에서 1인 부분 추출
#         object_region = orig_img * obj_mask[:, :, np.newaxis]

#         # 전처리 및 특성 추출
#         object_region = cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB)

#         # 이미지를 uint8로 변환
#         object_region = (object_region * 255).astype(np.uint8)

#         object_region = transform(object_region).unsqueeze(0).cuda()
#         features = model_siamese.forward_one(object_region)

#         # 특성 벡터를 텍스트 파일로 저장
#         output_filename = os.path.join(output_dir, f"feature_{obj_idx}.txt")
#         np.savetxt(output_filename, features.cpu().detach().numpy())

#     print(f"Object {i}: Extracted and saved {len(masks_data)} features")
