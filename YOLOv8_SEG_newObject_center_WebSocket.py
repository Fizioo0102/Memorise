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
from PIL import ImageFont, ImageDraw, Image
import asyncio
import websockets

# GPU 서버 설정
os.environ["CUDA_DEVICE_ORDER"]="j9b106"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# 모델 초기 설정
vector_size=512
fontpath = "C:/Users/SSAFY/Desktop/yolotest/fonts/NanumGothic.ttf"
font = ImageFont.truetype(fontpath, 20)

# MongoDB에 연결
client = MongoClient('mongodb://j9b106.p.ssafy.io:27017/')
db = client['test']
collection = db[f'yolo8_{vector_size}']

# FAISS 인덱스 초기화 및 MongoDB 데이터 로드
dimension = vector_size
index = faiss.IndexFlatL2(dimension)
faiss_db_ids = []  # DB ID 및 벡터 인덱스를 저장할 리스트

# 웹소켓 서버 설정
HOST = '70.12.130.111'  # 웹소켓 서버의 주소
PORT = 8765  # 웹소켓 서버의 포트

current_frame = None  # 현재 프레임을 저장하기 위한 변수

async def websocket_server(websocket, path):
    global current_frame
    while True:
        try:
            frame_data = await websocket.recv()
            # 데이터를 numpy array로 변환 (이 부분은 클라이언트에서 어떤 형태로 데이터를 전송하는지에 따라 수정이 필요합니다.)
            frame_np = ...  # 여기에 적절한 변환 로직을 넣어주세요.
            current_frame = frame_np
        except websockets.ConnectionClosed:
            print("Connection closed")
            break

# 웹소켓 서버 시작
start_server = websockets.serve(websocket_server, HOST, PORT)
asyncio.get_event_loop().run_until_complete(start_server)

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


# 탐지 중인 객체 마스크 색칠
def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image."""
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

flag=False

while True:
    if current_frame is None:
        continue

    frame = current_frame  # 웹소켓으로부터 받은 프레임을 사용

    results = model.predict(frame)
    print("--------------------------------------------------")
    print(results)

    # 화면 중앙 좌표 계산
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2

    closest_obj = None
    closest_distance = float('inf')

    for i, r in list(enumerate(results)):
        if r is None:
            cv2.imshow('New Object', frame)
            continue
        if r.masks is None:
            cv2.imshow('New Object', frame)
            continue    
        
        boxes_data = r.boxes.data.tolist()
        masks_data = r.masks.data
        orig_img = r.orig_img

        obj_idx=0
        obj_mask=[]

        for i, j in enumerate(masks_data):

            # 객체 중앙 좌표 계산
            x1, y1, x2, y2, _, _ = boxes_data[i]
            obj_center_x = (x1 + x2) / 2
            obj_center_y = (y1 + y2) / 2

            # 객체와 화면 중앙 사이의 거리 계산
            distance = np.sqrt((obj_center_x - center_x)**2 + (obj_center_y - center_y)**2)

            # 현재까지 가장 가운데에 위치한 객체인지 확인
            if distance < closest_distance:
                closest_distance = distance
                obj_idx=i
                obj_mask=j

            
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
        features_vector = features_roi_np.tolist()[0]  # 중첩 리스트 제거


        if flag == False:
            # 데이터베이스에 저장
            db_entry = {
                "vector": [features_vector]
            }
            result = collection.insert_one(db_entry)
            faiss_db_ids.append((result.inserted_id, 0))  # 새로 추가된 객체의 ID 및 벡터 인덱스 저장
            index.add(np.array(features_roi_np))
            flag = True
            # print(features_roi_np)


        else:
            # 실시간 영상에서 감지된 각 객체에 대해 가장 유사한 벡터를 faiss에서 검색
            D, I = index.search(features_roi_np, k=1)
            closest_obj_id, vector_idx = faiss_db_ids[I[0][0]]  # 벡터의 인덱스도 함께 가져옵니다.
            closest_obj = collection.find_one({"_id": closest_obj_id})

            # 가장 근접한 객체의 특성 벡터로 유사도 계산
            closest_obj_vector = torch.tensor(closest_obj['vector'][vector_idx]).cuda().unsqueeze(0)  # 해당 인덱스의 벡터를 가져옵니다.
            similarity = get_similarity(closest_obj_vector, features_roi, metric='cosine')  # 'euclidean' or 'cosine' or 'l1' or 'pearson'

            # 바운딩 박스 정보 가져오기
            x1, y1, x2, y2, _, _ = boxes_data[obj_idx]

            # 라벨 및 유사도 정보
            # label = f"Id: {closest_obj['_id']}, Index: {I[0][0]}, Sim: {similarity:.2f}"
            label = f"Id: {closest_obj['_id']}, Sim: {similarity:.2f}"
                    
            # 라벨을 그리는 위치 조정
            label_x = int(x1)
            label_y = int(y1 - 10)

            # 라벨 그리기
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Calculate overlay image with mask
            frame = overlay(orig_img, obj_mask, color=(0, 255, 0), alpha=0.3)

            

            # 유사도가 0.50 이상, 0.90 이하인 경우에만 벡터 추가
            if 0.70 <= similarity <= 0.95:
                # 유사도가 0.90 이상인 경우 해당 문서의 "vector" 필드에 새로운 벡터를 추가
                closest_obj['vector'].append(features_roi_np.tolist()[0])
                # MongoDB 업데이트
                collection.update_one({"_id": closest_obj_id}, {"$set": {"vector": closest_obj['vector']}})

                # faiss 변수 업데이트
                index.add(np.array(features_roi_np))
                faiss_db_ids.append((closest_obj_id, len(closest_obj['vector']) - 1))
                

        # Display the overlay image
        # cv2.imshow('Overlay', overlay_image)


        # cv2.imshow('New Object', frame)

         # 's' 키를 눌러 가중치 저장
        if cv2.waitKey(1) & 0xFF == ord('s'):
            torch.save(model_siamese.state_dict(), f'siamese_yolo8_{vector_size}_weights.pth')
            print("Weights saved!")

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        # # index의 사이즈 표시
        # index_size_text = f"등록 중 : {len(faiss_db_ids)/3:.1f} %"

        # # cv2.putText(frame, index_size_text, (center_x - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # frame = Image.fromarray(frame)
        # draw = ImageDraw.Draw(frame)
        # draw.text((center_x - 60, 30), index_size_text, (0,0,255), font=font)
        # frame = np.array(frame)

        # index의 사이즈 표시
        progress_percentage = len(faiss_db_ids) / 2  # 이 값은 0과 100 사이여야 합니다.
        max_bar_width = 200  # 프로그레스 바의 최대 너비를 설정합니다.
        bar_width = int(max_bar_width * progress_percentage / 100)  # 현재 퍼센트에 따른 프로그레스 바의 너비를 계산합니다.
        bar_height = 15  # 프로그레스 바의 높이를 설정합니다.
        start_x = center_x - max_bar_width // 2
        start_y = 30

        # 배경 바 그리기
        cv2.rectangle(frame, (start_x, start_y), (start_x + max_bar_width, start_y + bar_height), (255, 255, 255), 2)

        # 진행 상황에 따른 채워진 바 그리기
        cv2.rectangle(frame, (start_x, start_y), (start_x + bar_width, start_y + bar_height), (255, 106, 76), -1)  # -1은 내부를 채우는 것을 의미합니다.

        # 프로그레스 바 위에 텍스트 그리기
        progress_text = f"{progress_percentage:.1f} %"
        text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = start_y - 10  # 바의 중앙에 위치하도록 조정합니다.
        cv2.putText(frame, progress_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('New Object', frame)
    
cap.release()
cv2.destroyAllWindows()



