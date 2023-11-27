import cv2
import numpy as np
from rembg import remove
import torch

# 모델 로딩
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x')

# ORB 디스크립터를 초기화합니다.
orb = cv2.ORB_create()

# 로컬 이미지에서 특성점과 디스크립터를 추출합니다.
local_image = cv2.imread('C:/Users/SSAFY/Desktop/yolotest/imagetest3.jpg')  # 이미지 경로 입력
local_image = remove(local_image)
results_local = model_yolo(local_image)
if len(results_local.pred[0]) > 0:  # 객체가 탐지된 경우
    x1_local, y1_local, x2_local, y2_local = map(int, results_local.pred[0][0][:4])
    detected_local_obj = local_image[y1_local:y2_local, x1_local:x2_local]  # Detected object from local image
    cv2.imwrite("detected_object.jpg", detected_local_obj)  # Save the detected object

    kp_local, des_local = orb.detectAndCompute(detected_local_obj, None)

# FLANN 파라미터 설정
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)   # higher value gives better precision, it also takes more time
flann = cv2.FlannBasedMatcher(index_params, search_params)

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

        # 탐지된 객체에서 특성점과 디스크립터를 추출합니다.
        kp_roi, des_roi = orb.detectAndCompute(roi, None)

        if des_local is not None and des_roi is not None:
            matches = flann.knnMatch(des_local, des_roi, k=2)

            # 각 매치에서 두 개의 좋은 매치가 있는지 확인합니다.
    good_matches = [m1 for m in matches if len(m) == 2 for m1, m2 in [m] if m1.distance < 0.7 * m2.distance]

    label = f"Matches: {len(good_matches)}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    cv2.imshow('YOLOv5 Object Detection with ORB Matching', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
