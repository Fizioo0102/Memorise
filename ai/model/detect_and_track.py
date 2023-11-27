import torch
import cv2
import sys

# 모델 로딩
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # 'yolov5s'는 small 모델, 'yolov5m', 'yolov5l', 'yolov5x'도 선택 가능

# 웹캠에서 비디오 가져오기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO로 객체 탐지
    results = model(frame)
    
    # 결과를 프레임에 그리기
    rendered_frame = results.render()[0]
    
    cv2.imshow('YOLOv5 Object Detection', rendered_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
