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

# 특성 벡터 차원 조절
vector_size=512
# 몇 개의 객체 탐지할지 조절 - 화면 중앙에서 가까운 순으로
target_size=3

# MongoDB에 연결
client = MongoClient('mongodb://localhost:27017/')
db = client['test']
collection = db[f'yolo8_{vector_size}']

# FAISS 인덱스 초기화 및 MongoDB 데이터 로드
dimension = vector_size
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
        self.fc = nn.Linear(1408, vector_size)  # EfficientNetB2의 출력 차원을 1408로 수정

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

# 다른 거리 메트릭 시도 (예: L1 거리)
def l1_distance(features1, features2):
    return torch.norm(features1 - features2, p=1, dim=1).mean().item()

# 다른 유사도 메트릭 시도 (예: 피어슨 상관계수)
def pearson_correlation(features1, features2):
    mean1 = torch.mean(features1, dim=1)
    mean2 = torch.mean(features2, dim=1)
    cov = torch.mean((features1 - mean1) * (features2 - mean2), dim=1)
    std1 = torch.std(features1, dim=1)
    std2 = torch.std(features2, dim=1)
    return torch.mean(cov / (std1 * std2)).item()


# 유사도 및 거리 계산 함수 확장
# get_similarity 함수를 수정하여 다른 메트릭을 사용할 수 있도록 설정
def get_similarity(features1, features2, metric):
    if metric == 'cosine':
        return cosine_similarity(features1, features2)
    elif metric == 'euclidean':
        return -euclidean_distance(features1, features2)
    elif metric == 'l1':
        return -l1_distance(features1, features2)
    elif metric == 'pearson':
        return pearson_correlation(features1, features2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# YOLO8 모델 초기화
CONFIDENCE_THRESHOLD = 0.6
model = YOLO('yolov8x-seg.pt')
# Siamese Network 초기화 및 가중치 로드
model_siamese = SiameseNetwork().cuda()

# 체크포인트 파일이 존재하면 가중치를 로드합니다.
if os.path.exists(f'siamese_yolo8_{vector_size}_weights.pth'):
    model_siamese.load_state_dict(torch.load(f'siamese_yolo8_{vector_size}_weights.pth'))

model_siamese.eval()  # 평가 모드로 설정

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 웹캠 설정
cap = cv2.VideoCapture(0)
# cap.set(3, 1920)
# cap.set(4, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.flip(frame, 1)  # 화면 좌우 반전

    results = model.predict(frame)

    # 화면 중앙 좌표 계산
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2

    closest_objs = []  # 화면 중앙에서 가장 가까운 객체 저장 리스트
    closest_distances = []  # 화면 중앙에서 가장 가까운 객체들과의 거리 저장 리스트

    for i, r in list(enumerate(results)):
        if r is None:
            cv2.imshow('YOLOv8 Object Detection', frame)
            continue
        if r.masks is None:
            cv2.imshow('YOLOv8 Object Detection', frame)
            continue    
        
        boxes_data = r.boxes.data.tolist()
        masks_data = r.masks.data
        orig_img = r.orig_img

        for obj_idx, obj_mask in enumerate(masks_data):
            obj_mask = obj_mask.cpu().numpy()

            # 객체 중앙 좌표 계산
            x1, y1, x2, y2, _, _ = boxes_data[obj_idx]
            obj_center_x = (x1 + x2) / 2
            obj_center_y = (y1 + y2) / 2

            # 객체와 화면 중앙 사이의 거리 계산
            distance = np.sqrt((obj_center_x - center_x)**2 + (obj_center_y - center_y)**2)

            # 현재까지 가장 가운데에 위치한 객체인지 확인
            if len(closest_objs) < target_size:
                closest_objs.append((obj_idx, obj_mask))
                closest_distances.append(distance)
            else:
                # 현재 객체가 가장 가까운 객체보다 더 가까울 경우, 가장 가까운 객체들 중 가장 먼 객체를 대체
                max_distance_idx = np.argmax(closest_distances)
                if distance < closest_distances[max_distance_idx]:
                    closest_objs[max_distance_idx] = (obj_idx, obj_mask)
                    closest_distances[max_distance_idx] = distance

        

        for obj_idx, obj_mask in closest_objs:
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
            similarity = get_similarity(closest_obj_vector, features_roi, metric='cosine')  # 'euclidean' or 'cosine' or 'l1' or 'pearson'


            # 유사도가 0.90 이상인 경우에만 라벨 표시
            if similarity >= 0.50:
                # 바운딩 박스 정보 가져오기
                x1, y1, x2, y2, _, _ = boxes_data[obj_idx]

                # 라벨 및 유사도 정보
                # label = f"Id: {closest_obj['_id']}, Index: {I[0][0]}, Sim: {similarity:.2f}"
                label = f"Index: {I[0][0]}, Sim: {similarity:.2f}"
                
                # 라벨을 그리는 위치 조정
                label_x = int(x1)
                label_y = int(y1 - 10)

                # 라벨 그리기
                cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)    

                if 0.95 <= similarity <= 0.95:
                    # 유사도가 0.90 이상인 경우 해당 문서의 "vector" 필드에 새로운 벡터를 추가
                    closest_obj['vector'].append(features_roi_np.tolist()[0])

                    # MongoDB 업데이트
                    collection.update_one({"_id": closest_obj_id}, {"$set": {"vector": closest_obj['vector']}})

                    index.add(np.array(features_roi_np))
                    faiss_db_ids.append((closest_obj_id, len(closest_obj['vector']) - 1))

    
        cv2.imshow('YOLOv8 Object Detection', frame)


         # 's' 키를 눌러 가중치 저장
        if cv2.waitKey(1) & 0xFF == ord('s'):
            torch.save(model_siamese.state_dict(), f'siamese_yolo8_{vector_size}_weights.pth')
            print("Weights saved!")

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()



