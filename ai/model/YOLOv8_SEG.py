import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pymongo import MongoClient
import faiss
import timm

# MongoDB에 연결
client = MongoClient('mongodb://localhost:27017/')
db = client['test']
collection = db['yolo8']

# FAISS 인덱스 초기화 및 MongoDB 데이터 로드
dimension = 512
index = faiss.IndexFlatL2(dimension)
objects_in_db = list(collection.find({}))
vectors = [vec for obj in objects_in_db for vec in obj['vector']]  # 모든 벡터를 한 리스트에 추가합니다.
faiss_db_ids = [(obj['_id'], idx) for obj in objects_in_db for idx, _ in enumerate(obj['vector'])]  # 각 벡터에 대한 DB ID와 벡터의 인덱스를 함께 저장합니다.

if vectors:
    index.add(np.array(vectors))

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


# 여러 유사도 및 거리 메트릭을 위한 유틸리티 함수 정의
def cosine_similarity(features1, features2):
    return nn.functional.cosine_similarity(features1, features2, dim=1).mean().item()

def euclidean_distance(features1, features2):
    return torch.norm(features1 - features2, dim=1).mean().item()


# 유사도 및 거리 계산 함수 확장
def get_similarity(features1, features2, metric='cosine'):
    if metric == 'cosine':
        return cosine_similarity(features1, features2)
    elif metric == 'euclidean':
        return -euclidean_distance(features1, features2)  # '-' 를 붙여 거리가 짧을수록 값이 높게 나오도록 조정
    else:
        raise ValueError(f"Unknown metric: {metric}")


# YOLO 모델 초기화
CONFIDENCE_THRESHOLD = 0.6
model = YOLO('yolov8x-seg.pt')
# Siamese Network 초기화 및 가중치 로드
model_siamese = SiameseNetwork().cuda()

# 체크포인트 파일이 존재하면 가중치를 로드합니다.
if os.path.exists('siamese_yolo8_weights.pth'):
    model_siamese.load_state_dict(torch.load('siamese_yolo8_weights.pth'))

model_siamese.eval()  # 평가 모드로 설정

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# results = model.predict(source="0", save=True, show=True, stream=True)


# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 여기서 frame을 좌우 반전시킵니다.

    results = model.predict(frame)

    # print(list(enumerate(results)))

    for i, r in list(enumerate(results)):
        if r is None:
            continue
        
        boxes_data = r.boxes.data.tolist()
        masks_data = r.masks.data
        orig_img = r.orig_img

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
            features_roi = model_siamese.forward_one(object_region)

            features_roi_np = features_roi.cpu().detach().numpy()

            # 실시간 영상에서 감지된 각 객체에 대해 가장 유사한 벡터를 faiss에서 검색
            D, I = index.search(features_roi_np, k=1)
            closest_obj_id, vector_idx = faiss_db_ids[I[0][0]]  # 벡터의 인덱스도 함께 가져옵니다.
            closest_obj = collection.find_one({"_id": closest_obj_id})

            # 가장 근접한 객체의 특성 벡터로 유사도 계산
            closest_obj_vector = torch.tensor(closest_obj['vector'][vector_idx]).cuda().unsqueeze(0)  # 해당 인덱스의 벡터를 가져옵니다.
            similarity = get_similarity(closest_obj_vector, features_roi, metric='cosine')  # 'euclidean' or 'cosine'

            # 바운딩 박스 정보 가져오기
            x1, y1, x2, y2, _, _ = boxes_data[obj_idx]

            # 라벨 및 유사도 정보
            label = f"Name: {closest_obj['name']}, Index: {I[0][0]}, Sim: {similarity:.2f}"
            
            # 라벨을 그리는 위치 조정
            label_x = int(x1)
            label_y = int(y1 - 10)

            # 라벨 그리기
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imshow('YOLOv8 Object Detection', frame)


         # 예를 들어 's' 키를 눌러 가중치를 저장하게 만들 수 있습니다.
        if cv2.waitKey(1) & 0xFF == ord('s'):
            torch.save(model_siamese.state_dict(), 'siamese_yolo8_weights.pth')
            print("Weights saved!")

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()



