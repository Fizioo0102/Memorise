import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from ultralytics import YOLO
from pymongo import MongoClient
import faiss
import timm

import argparse
import asyncio
import json
import logging
import os
import platform

from aiohttp import web
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
import ssl
import urllib

use_cuda = torch.cuda.is_available()


def send_data(channel, data):
    """
    Send data to the client through the WebRTC data channel.
    
    Args:
        channel: The WebRTC data channel.
        data: The data to send (typically a dictionary).
    """
    channel.send(json.dumps(data))


# 특성 벡터 차원 조절
vector_size=512
# 몇 개의 객체 탐지할지 조절 - 화면 중앙에서 가까운 순으로
target_size=1

# MongoDB에 연결
username = urllib.parse.quote_plus("jylee")
password = urllib.parse.quote_plus("memorise@106@jyjy@account")
host = "j9b106.p.ssafy.io"
port = "27017"
database = "memorise"
authDB = "admin"

connection_string = f"mongodb://{username}:{password}@{host}:{port}/{database}?authSource={authDB}"
client = MongoClient(connection_string)
db = client['memorise']
collection = db[f'yolo8_{vector_size}']

# FAISS 인덱스 초기화 및 MongoDB 데이터 로드
dimension = vector_size
faiss_index = faiss.IndexFlatL2(dimension)
objects_in_db = list(collection.find({}))
vectors = [vec for obj in objects_in_db for vec in obj['vector']]  # 모든 벡터를 한 리스트에 추가합니다.
faiss_db_ids = [(obj['_id'], idx) for obj in objects_in_db for idx, _ in enumerate(obj['vector'])]  # 각 벡터에 대한 DB ID와 벡터의 인덱스를 함께 저장합니다.

if vectors:
    faiss_index.add(np.array(vectors))

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_network = timm.create_model("tf_efficientnet_b2.ns_jft_in1k", pretrained=True)
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

if use_cuda:
    model_siamese = model_siamese.cuda()

# 2. YOLO 모델을 GPU로 이동 (만약 YOLO가 GPU를 지원한다면)
if hasattr(model, 'cuda') and use_cuda:
    model = model.cuda()

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


relay = MediaRelay()


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track
        self.frame_counter = 0  # <-- 프레임 카운터 초기화

    async def recv(self):
        frame = await self.track.recv()

        # 프레임 카운터 업데이트 및 출력
        self.frame_counter += 1
        print(f"Processing frame {self.frame_counter}")  # <-- 현재 프레임 번호 출력

        # Convert AIORTC frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        if self.frame_counter%2!=0:

            results = model.predict(img)

            # 화면 중앙 좌표 계산
            center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

            closest_objs = []  # 화면 중앙에서 가장 가까운 객체 저장 리스트
            closest_distances = []  # 화면 중앙에서 가장 가까운 객체들과의 거리 저장 리스트

            for i, r in list(enumerate(results)):
                if r is None:
                    # cv2.imshow('YOLOv8 Object Detection', img)
                    continue
                if r.masks is None:
                    # cv2.imshow('YOLOv8 Object Detection', img)
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
                    D, I = faiss_index.search(features_roi_np, k=1)
                    closest_obj_id, vector_idx = faiss_db_ids[I[0][0]]  # 벡터의 인덱스도 함께 가져옵니다.
                    closest_obj = collection.find_one({"_id": closest_obj_id})

                    # 가장 근접한 객체의 특성 벡터로 유사도 계산
                    closest_obj_vector = torch.tensor(closest_obj['vector'][vector_idx]).cuda().unsqueeze(0)  # 해당 인덱스의 벡터를 가져옵니다.
                    similarity = get_similarity(closest_obj_vector, features_roi, metric='cosine')  # 'euclidean' or 'cosine' or 'l1' or 'pearson'


                    # 유사도가 0.50 이상인 경우에만 라벨 표시
                    if similarity >= 0.50:
                        # 바운딩 박스 정보 가져오기
                        x1, y1, x2, y2, _, _ = boxes_data[obj_idx]

                        # 라벨 및 유사도 정보
                        label = f"Id: {closest_obj['_id']}, Sim: {similarity:.2f}"
                        # label = f"Index: {I[0][0]}, Sim: {similarity:.2f}"
                        
                        # 라벨을 그리는 위치 조정
                        label_x = int(x1)
                        label_y = int(y1 - 10)

                        data_to_send = {
                            "id": str(closest_obj['_id']),  # ObjectId를 문자열로 변환
                            "label_x": label_x,
                            "label_y": label_y
                        }
                        print("좌표 : ", data_to_send)
                        send_data(data_channel, data_to_send)  # 데이터 채널을 통해 클라이언트로 데이터 전송

  

                        if 0.95 <= similarity <= 0.95:
                            # 유사도가 0.90 이상인 경우 해당 문서의 "vector" 필드에 새로운 벡터를 추가
                            closest_obj['vector'].append(features_roi_np.tolist()[0])

                            # MongoDB 업데이트
                            collection.update_one({"_id": closest_obj_id}, {"$set": {"vector": closest_obj['vector']}})

                            faiss_index.add(np.array(features_roi_np))
                            faiss_db_ids.append((closest_obj_id, len(closest_obj['vector']) - 1))

        # Show frame using OpenCV
        # cv2.imshow("Client Stream", img)
        # cv2.waitKey(1)

        return frame



async def index(request):
    content = open(os.path.join("index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join("client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % id(pc)

    def log_info(msg, *args):
        logging.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        global data_channel  # global 변수로 설정
        data_channel = channel  # 채널 초기화


    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            pc.addTrack(VideoTransformTrack(
                relay.subscribe(track)
            ))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


pcs = set()


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8082, help="Port for HTTP server (default: 8080)"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)

