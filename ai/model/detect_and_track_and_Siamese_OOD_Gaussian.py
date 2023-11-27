import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
# from rembg import remove
from pymongo import MongoClient
import faiss
from sklearn.mixture import GaussianMixture

# GMM 모델 초기화
gmm = GaussianMixture(n_components = 2) # 예시 : 2개의 가우시안 컴포넌트로 초기화

# MongoDB에 연결
client = MongoClient('mongodb://localhost:27017/')
db = client['test']
collection = db['object']

# FAISS 인덱스 초기화 및 MongoDB 데이터 로드
dimension = 256  # SiameseNetwork에서 출력되는 벡터의 차원
index = faiss.IndexFlatL2(dimension)
objects_in_db = list(collection.find({}))
vectors = [obj['vector'] for obj in objects_in_db]
faiss_db_ids = [obj['_id'] for obj in objects_in_db]
index.add(np.array(vectors))


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_network = models.resnet50(pretrained=True)
        self.fc = nn.Linear(1000, 256)
        

    def forward_one(self, x):
        x = self.base_network(x)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# 유사도 계산
def get_similarity(features1, features2):
    cos_sim = nn.functional.cosine_similarity(features1, features2, dim=1)
    return cos_sim.mean()

# 모델 로딩
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x')
# Siamese Network 초기화 및 가중치 로드
model_siamese = SiameseNetwork().cuda()

# 체크포인트 파일이 존재하면 가중치를 로드합니다.
if os.path.exists('siamese_weights.pth'):
    model_siamese.load_state_dict(torch.load('siamese_weights.pth'))

model_siamese.eval()  # 평가 모드로 설정
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 로컬 이미지 처리
local_image = cv2.imread('C:/Users/SSAFY/Desktop/yolotest/cup2.jpg')
# local_image = remove(local_image)
local_image_rgb = cv2.cvtColor(local_image, cv2.COLOR_BGR2RGB)

# 로컬 이미지에서 YOLO로 객체 탐지
results_local = model_yolo(local_image_rgb)
x1_local, y1_local, x2_local, y2_local = map(int, results_local.pred[0][0][:4])
detected_local_obj = local_image_rgb[y1_local:y2_local, x1_local:x2_local]  # Detected object from local image

tensor_local_obj = transform(detected_local_obj).unsqueeze(0).cuda()

# 로컬 이미지의 객체 특성 벡터 변환 및 출력
features_local_obj = model_siamese.forward_one(tensor_local_obj)
numpy_array_features = features_local_obj.cpu().detach().numpy()  # Tensor -> NumPy array
list_features = numpy_array_features.tolist()  # NumPy array -> Python list
print("Features of detected local object (for MongoDB):", list_features)

# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 실시간 영상에서 YOLO로 객체 탐지
    results = model_yolo(frame)

    for *xyxy, conf, cls in results.pred[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        roi = frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        tensor_roi = transform(roi_rgb).unsqueeze(0).cuda()
        
        # 실시간 영상에서 객체 특성 벡터 변환
        features_roi = model_siamese.forward_one(tensor_roi)
        features_roi_np = features_roi.cpu().detach().numpy()

        # GMM 모델로 확률 예측
        ood_probability = gmm.score_samples(features_roi_np)[0]

        if ood_probability < threshold:
            continue
        
        # 실시간 영상에서 감지된 각 객체에 대해 가장 유사한 벡터를 faiss에서 검색
        D, I = index.search(features_roi_np, k=1)
        closest_obj_id = faiss_db_ids[I[0][0]]
        closest_obj = collection.find_one({"_id": closest_obj_id})

        # 가장 근접한 객체의 특성 벡터로 유사도 계산
        closest_obj_vector = torch.tensor(closest_obj['vector']).cuda().unsqueeze(0)

        similarity = get_similarity(closest_obj_vector, features_roi).item()

        label = f"Name: {closest_obj['name']}, Sim: {similarity:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    cv2.imshow('YOLOv5 Object Detection', frame)

    
     # 예를 들어 's' 키를 눌러 가중치를 저장하게 만들 수 있습니다.
    if cv2.waitKey(1) & 0xFF == ord('s'):
        torch.save(model_siamese.state_dict(), 'siamese_weights.pth')
        print("Weights saved!")

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()