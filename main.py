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
import aiohttp
from aiohttp import ClientSession, web, FormData
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
import ssl
import urllib
from scipy import ndimage
import threading
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

use_cuda = torch.cuda.is_available()

# 특성 벡터 차원 조절
vector_size=256
# 몇 개의 객체 탐지할지 조절
target_size=2
# 탐지 임계값
CONFIDENCE_THRESHOLD = 0.4

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

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_network = timm.create_model("tf_efficientnet_b7_ns", pretrained=True)
        self.fc = nn.Linear(2560, vector_size)  # EfficientNetB2의 출력 차원을 1408로 수정

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


# 유사도 및 거리 계산 함수 확장
# get_similarity 함수를 수정하여 다른 메트릭을 사용할 수 있도록 설정
def get_similarity(features1, features2, metric):
    if metric == 'cosine':
        return cosine_similarity(features1, features2)


# YOLO8 모델 초기화
model = YOLO('yolov8l-seg.pt')

# Siamese Network 초기화 및 가중치 로드
model_siamese = SiameseNetwork().cuda()

if use_cuda:
    model_siamese = model_siamese.cuda()

# YOLO 모델을 GPU로 이동 (만약 YOLO가 GPU를 지원한다면)
if hasattr(model, 'cuda') and use_cuda:
    model = model.cuda()

# 체크포인트 파일이 존재하면 가중치를 로드합니다.
if os.path.exists(f'siamese_tf_efficientnet_b7_ns_{vector_size}_weights.pth'):
    model_siamese.load_state_dict(torch.load(f'siamese_tf_efficientnet_b7_ns_{vector_size}_weights.pth'))

model_siamese.eval()  # 평가 모드로 설정

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
relay = MediaRelay()

def process_object_region(obj_mask, orig_img, x1, x2, y1, y2):
    # 원시 마스크의 크기를 객체 이미지의 크기에 맞게 조절
    obj_mask_resized = cv2.resize(obj_mask, (orig_img.shape[1], orig_img.shape[0]))
    # 원시 마스크에서 1인 부분 추출
    object_region = orig_img * obj_mask_resized[:, :, np.newaxis]

    # 객체 영역을 자릅니다.
    cropped_object_region = object_region[int(y1):int(y2), int(x1):int(x2)]

    # 전처리 및 특성 추출
    object_region_rgb = cv2.cvtColor(cropped_object_region, cv2.COLOR_BGR2RGB)
    object_region_uint8 = (object_region_rgb * 255).astype(np.uint8)
    object_region_transformed = transform(object_region_uint8).unsqueeze(0)

    # # 여기에서 디렉토리를 생성합니다.
    # os.makedirs('test_images', exist_ok=True)
    # # 회전된 이미지를 로컬에 저장
    # cv2.imwrite(f'test_images/image_{x1}_angle_{y1}.jpg', cv2.cvtColor(object_region_rgb, cv2.COLOR_BGR2RGB))



    if use_cuda:  # 텐서를 GPU로 이동
        object_region_transformed = object_region_transformed.cuda()
    return object_region_transformed

# 회전 이미지 여백 제거
def process_rotated_image(rotated_image):
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    thresh = cv2.convertScaleAbs(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]

    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[0]+box[2] for box in boxes)
    y_max = max(box[1]+box[3] for box in boxes)
    
    cropped_image = rotated_image[y_min:y_max, x_min:x_max]
    return cropped_image

# def process_images_and_vectors(image_list, model_siamese, transform, use_cuda, result_id):
#     vector_list=[]
#     for idx, image in enumerate(image_list):
#         for angle in [45, 90, 135, 180, 225, 270, 315]:
#             rotated_image = ndimage.rotate(image, angle, reshape=True)
#             rotated_image = process_rotated_image(rotated_image)  # 까만 영역을 제거한 이미지를 얻습니다.
#             rotated_image_uint8 = (rotated_image * 255).astype(np.uint8)

#             rotated_image_transformed = transform(rotated_image_uint8).unsqueeze(0)
#             if use_cuda:
#                 rotated_image_transformed = rotated_image_transformed.cuda()
#             rotated_features_roi = model_siamese.forward_one(rotated_image_transformed)
#             rotated_features_roi_np = rotated_features_roi.cpu().detach().numpy()
#             rotated_features_vector = rotated_features_roi_np.tolist()[0]  # 중첩 리스트 제거
#             vector_list.append(rotated_features_vector)  # 회전된 이미지의 벡터를 vector_list에 추가합니다.

#     # MongoDB 업데이트
#     collection.update_one({'_id': result_id}, {'$push': {'vector': {'$each': vector_list}}})

def process_images_and_vectors(image_list, model_siamese, transform, use_cuda, result_id):
    vector_list=[]
    for idx, image in enumerate(image_list):
        for angle in [10, 20, 30, 40, 50, 60, 70 , 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]:
            rotated_image = ndimage.rotate(image, angle, reshape=True)
            rotated_image_crop = process_rotated_image(rotated_image)  # Remove black areas

            # # Get the dimensions of the cropped image
            # h, w = rotated_image_crop.shape[:2]
            
            # # Determine the new dimensions and padding
            # target_w, target_h = 352, 640
            # aspect_ratio = w / h

            # if aspect_ratio > (target_w / target_h):
            #     new_w = target_w
            #     new_h = int(target_w / aspect_ratio)
            # else:
            #     new_h = target_h
            #     new_w = int(target_h * aspect_ratio)

            # pad_top = (target_h - new_h) // 2
            # pad_bottom = target_h - new_h - pad_top
            # pad_left = (target_w - new_w) // 2
            # pad_right = target_w - new_w - pad_left

            # # Resize the image while maintaining the aspect ratio
            # resized_image = cv2.resize(rotated_image_crop, (new_w, new_h))

            # Add padding to meet the required dimensions
            # padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # Convert the padded image to uint8
            padded_image_uint8 = (rotated_image_crop * 255).astype(np.uint8)

            # # 여기에서 디렉토리를 생성합니다.
            # os.makedirs('local_images', exist_ok=True)
            # # 회전된 이미지를 로컬에 저장
            # cv2.imwrite(f'local_images/image_{idx}_angle_{angle}.jpg', cv2.cvtColor(rotated_image_crop, cv2.COLOR_RGB2BGR))

            # Transform and process the image as in the original code
            padded_image_transformed = transform(padded_image_uint8).unsqueeze(0)
            if use_cuda:
                padded_image_transformed = padded_image_transformed.cuda()
            rotated_features_roi = model_siamese.forward_one(padded_image_transformed)
            rotated_features_roi_np = rotated_features_roi.cpu().detach().numpy()
            rotated_features_vector = rotated_features_roi_np.tolist()[0]  # Remove nested list
            vector_list.append(rotated_features_vector)  # Add the vector of the rotated image to vector_list

    # MongoDB 업데이트
    collection.update_one({'_id': result_id}, {'$push': {'vector': {'$each': vector_list}}})

def async_process_images_and_vectors(image_list, model_siamese, transform, use_cuda, result_id):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_in_executor(None, process_images_and_vectors, image_list, model_siamese, transform, use_cuda, result_id)
    loop.run_forever()


def send_data(channel, data):
    """
    Send data to the client through the WebRTC data channel.
    Args:
        channel: The WebRTC data channel.
        data: The data to send (typically a dictionary).
    """
    channel.send(json.dumps(data))

class VideoTransformTrack2(VideoStreamTrack):
    async def send_post_request(self, url, headers, data):
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                return await response.text()

    async def send_file_post_request(self, url, file):
        data = FormData()
        data.add_field('file', file, filename='frame.jpeg', content_type='image/jpeg')
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.json()  # 응답이 JSON 형식이라고 가정


    def __init__(self, track, session):
        super().__init__()  # don't forget this!
        self.track = track
        self.session = session  # <-- session 저장
        self.frame_counter = 0  # 프레임 카운터 초기화
        self.first_object = None

        # FAISS 인덱스 초기화 및 MongoDB 데이터 로드
        self.dimension = vector_size
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.faiss_db_ids = []  # DB ID 및 벡터 인덱스를 저장할 리스트
        self.vector_list = []  # 벡터를 임시 저장할 리스트
        self.image_list = []  # 객체 이미지 저장할 리스트 (회전용)

        self.end_check = False
        
    async def recv(self):
        frame = await self.track.recv()
        self.frame_counter += 1
        print(f"Processing frame {self.frame_counter}")

        if self.frame_counter%6==0:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (352, 640))

            results = model.predict(img, conf=CONFIDENCE_THRESHOLD, max_det=1, stream_buffer=False, retina_masks=True, half=False, device=0)

            if results:
                center_x, center_y = 192, 378
                closest_obj = None
                closest_distance = float('inf')

                for i, r in list(enumerate(results)):
                    if r is None or r.masks is None:
                        continue

                    boxes_data = r.boxes.data.tolist()
                    masks_data = r.masks.data
                    orig_img = r.orig_img

                    obj_idx = 0
                    obj_mask = []

                    for i, j in enumerate(masks_data):
                        x1, y1, x2, y2, _, _ = boxes_data[i]
                        obj_center_x = (x1 + x2) / 2
                        obj_center_y = (y1 + y2) / 2
                        distance = np.sqrt((obj_center_x - center_x)**2 + (obj_center_y - center_y)**2)

                        if distance < closest_distance:
                            closest_distance = distance
                            obj_idx = i
                            obj_mask = j

                    obj_mask = obj_mask.cpu().numpy()
                    # 원시 마스크의 크기를 객체 이미지의 크기에 맞게 조절
                    obj_mask = cv2.resize(obj_mask, (orig_img.shape[1], orig_img.shape[0]))

                    # 원시 마스크에서 1인 부분 추출
                    object_region = orig_img * obj_mask[:, :, np.newaxis]

                    # 객체 영역의 바운딩 박스를 찾습니다.
                    x1, y1, x2, y2, _, _ = boxes_data[obj_idx]

                    # 객체 영역을 자릅니다.
                    cropped_object_region = object_region[int(y1):int(y2), int(x1):int(x2)]

                    # 전처리 및 특성 추출
                    object_region_rgb = cv2.cvtColor(cropped_object_region, cv2.COLOR_BGR2RGB)

                    # 이미지를 uint8로 변환
                    object_region = (object_region_rgb * 255).astype(np.uint8)

                    object_region = transform(object_region).unsqueeze(0).cuda()
                    features_roi = model_siamese.forward_one(object_region)

                    features_roi_np = features_roi.cpu().detach().numpy()
                    features_vector = features_roi_np.tolist()[0]  # 중첩 리스트 제거

                    # 첫 번째 프레임인 경우 벡터 리스트에 추가 
                    if self.frame_counter == 6:
                        self.vector_list.append(features_roi_np.tolist()[0])
                        # object_region_rgb를 image_list에 추가합니다.
                        self.image_list.append(object_region_rgb)
                        self.first_object=object_region_rgb
                    else:
                        # 가장 유사한 벡터와의 유사도 계산
                        similarities = [get_similarity(torch.tensor(vec).cuda().unsqueeze(0), features_roi, metric='cosine') 
                                        for vec in self.vector_list]
                        
                        print(len(self.vector_list)," : ", max(similarities))

                        # 유사도가 0.70 이상 0.97 이하인 경우 벡터 리스트에 추가
                        if 0.70 <= max(similarities) <= 0.95:
                            self.vector_list.append(features_roi_np.tolist()[0])
                            # object_region_rgb를 image_list에 추가합니다.
                            self.image_list.append(object_region_rgb)  

                    # 벡터 리스트에 100개의 벡터가 저장되면 MongoDB에 insert
                    if len(self.vector_list) == 10 and self.end_check == False:
                        db_entry = {
                            "vector": self.vector_list
                        }
                        result = collection.insert_one(db_entry)
                        self.faiss_db_ids.extend([(result.inserted_id, idx) for idx in range(100)])
                        self.faiss_index.add(np.array(self.vector_list))

                        # 비동기로 이미지 및 벡터 처리 함수 호출
                        threading.Thread(target=async_process_images_and_vectors, args=(self.image_list, model_siamese, transform, use_cuda, result.inserted_id)).start()
                        
                        # 첫 번째 프레임을 바이트 버퍼로 변환
                        _, encoded_image = cv2.imencode('.jpeg', self.first_object)
                        byte_buffer = encoded_image.tobytes()

                        # 지정된 엔드포인트로 파일 전송
                        file_upload_response = await self.send_file_post_request(
                            "http://j9b106.p.ssafy.io:8000/items/upload", byte_buffer)

                        # 파일 업로드 응답에서 'savedFileName' 사용
                        saved_file_name = file_upload_response.get('savedFileName')

                        # 'savedFileName'을 이용하여 이미지 URL 구성
                        image_url = f"https://b106-memorise.s3.ap-northeast-2.amazonaws.com/{saved_file_name}"

                        # 이미지 URL을 사용하여 JSON POST 요청 전송
                        json_post_url = "http://j9b106.p.ssafy.io:8000/items"
                        json_post_headers = {"Content-Type": "application/json"}
                        json_post_data = {
                            "itemName": str(result.inserted_id),
                            "itemImage": image_url  # 여기에 구성된 이미지 URL 사용
                        }
                        json_post_response_text = await self.send_post_request(
                            json_post_url, json_post_headers, json_post_data)
                        
                        print(json_post_response_text)

                        data_to_send = {
                            "count": len(self.vector_list)
                        }
                        send_data(self.session.data_channel, data_to_send)

                        data_to_send = {
                            "newId": str(result.inserted_id)
                        }
                        send_data(self.session.data_channel, data_to_send)

                        self.end_check = True
                    
                    else:
                        data_to_send = {
                            "count": len(self.vector_list)
                        }
                        send_data(self.session.data_channel, data_to_send)
        return frame

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, session):
        super().__init__()
        self.track = track
        self.session = session  # <-- session 저장
        self.frame_counter = 0  # <-- 프레임 카운터 초기화

        # FAISS 인덱스 초기화 및 MongoDB 데이터 로드    
        self.dimension = vector_size
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.objects_in_db = list(collection.find({}))

        # 모든 벡터를 한 리스트에 추가하고, 각 벡터에 대한 DB ID와 벡터의 인덱스를 함께 저장
        self.vectors = [vec for obj in self.objects_in_db for vec in obj['vector']]
        self.faiss_db_ids = [(obj['_id'], idx) for obj in self.objects_in_db for idx, _ in enumerate(obj['vector'])]

        # FAISS 인덱스에 벡터 추가
        if self.vectors:
            self.faiss_index.add(np.array(self.vectors))

    async def recv(self):
        frame = await self.track.recv()

        # 프레임 카운터 업데이트 및 출력
        self.frame_counter += 1
        print(f"Processing frame {self.frame_counter}")  # <-- 현재 프레임 번호 출력
        print(f"Frame dimensions: {frame.width}x{frame.height}")

        if self.frame_counter%5==0:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (352, 640))

            results = model.predict(img, conf=CONFIDENCE_THRESHOLD, max_det=target_size, stream_buffer=False, retina_masks=True, half=False, device=0)

            for i, r in list(enumerate(results)):
                if r is None:
                    continue
                if r.masks is None:
                    continue
                
                
                boxes_data = r.boxes.data.tolist()
                masks_data = r.masks.data


                objects_data = []  # 모든 객체의 데이터를 담을 리스트를 초기화합니다.
                for obj_idx, obj_mask in enumerate(masks_data):
                    obj_mask = obj_mask.cpu().numpy()
                    # 객체 영역의 바운딩 박스를 찾습니다.
                    x1, y1, x2, y2, _, _ = boxes_data[obj_idx]


                    # 원시 마스크의 크기를 객체 이미지의 크기에 맞게 조절
                    features_roi = model_siamese.forward_one(process_object_region(obj_mask, r.orig_img, x1, x2, y1, y2))
                    features_roi_np = features_roi.cpu().detach().numpy()

                    # 실시간 영상에서 감지된 각 객체에 대해 가장 유사한 벡터를 faiss에서 검색
                    D, I = self.faiss_index.search(features_roi_np, k=1)
                    closest_obj_id, vector_idx = self.faiss_db_ids[I[0][0]]  # 벡터의 인덱스도 함께 가져옵니다.
                    
                    # 이미 로드된 objects_in_db에서 해당 객체를 찾습니다.
                    closest_obj = next(obj for obj in self.objects_in_db if obj['_id'] == closest_obj_id)

                    # 가장 근접한 객체의 특성 벡터로 유사도 계산
                    closest_obj_vector = torch.tensor(closest_obj['vector'][vector_idx])  # 해당 인덱스의 벡터를 가져옵니다.
                    if use_cuda:  # 텐서를 GPU로 이동
                        closest_obj_vector = closest_obj_vector.cuda()
                    closest_obj_vector = closest_obj_vector.unsqueeze(0)
                    similarity = get_similarity(closest_obj_vector, features_roi, metric='cosine')  # 'euclidean' or 'cosine' or 'l1' or 'pearson'

                    # 바운딩 박스 정보 가져오기
                    x1, y1, x2, y2, _, _ = boxes_data[obj_idx]
                    
                    # 객체 중앙 좌표 구하기
                    label_x = int((x1 + x2) // 2)
                    label_y = int((y1 + y2) // 2)

                    print(similarity)

                    # 유사도가 0.50 이상인 경우에만 아이디 전달
                    if similarity >= 0.80:
                        object_data = {
                            "id": str(closest_obj['_id']),  # ObjectId를 문자열로 변환
                            "label_x": label_x,
                            "label_y": label_y,
                            # "width": int(x2 - x1),
                            # "height": int(y2 - y1)
                        }
                    else:
                        object_data = {
                            "id": "0",  # ObjectId를 문자열로 변환
                            "label_x": label_x,
                            "label_y": label_y,
                            # "width": int(x2 - x1),
                            # "height": int(y2 - y1)
                        }

                    objects_data.append(object_data)  # 각 객체의 데이터를 리스트에 추가합니다.

                data_to_send = {
                    "objects": objects_data  # 객체 데이터의 리스트를 보냅니다.
                }
                print(data_to_send)
                send_data(self.session.data_channel, data_to_send)  # <-- 수정된 부분

                
        return frame


class ClientSession:
    def __init__(self, pc):
        self.pc = pc  # RTCPeerConnection 인스턴스
        self.data_channel = None  # DataChannel 인스턴스

        
# 클라이언트 세션을 저장할 글로벌 딕셔너리
client_sessions = {}


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    track_type = params.get("trackType", "default")  # 기본값은 "default"

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % id(pc)

    session = ClientSession(pc)  # 새로운 클라이언트 세션 생성
    client_sessions[pc_id] = session  # 클라이언트 세션 저장

    def log_info(msg, *args):
        logging.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        session.data_channel = channel  # 클라이언트 세션의 데이터 채널 설정


    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            client_sessions.pop(pc_id, None)  # <-- 세션을 client_sessions에서 제거

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            if track_type == "track2":
                pc.addTrack(VideoTransformTrack2(
                    relay.subscribe(track), session  # <-- session 인자 전달
                ))
            else:  # 기본값은 VideoTransformTrack 사용
                pc.addTrack(VideoTransformTrack(
                    relay.subscribe(track), session  # <-- session 인자 전달
                ))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await pc.close()
            pcs.discard(pc)
            client_sessions.pop(pc_id, None)  # <-- 세션을 client_sessions에서 제거

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
    client_sessions.clear()

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
    app.router.add_post("/offer", offer)
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)