import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from rembg import remove

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
local_image = cv2.imread('C:/Users/SSAFY/Desktop/yolotest/jun2.jpg')
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

# 1. Python list를 NumPy array로 변환
numpy_array_back = np.array(list_features)
# 2. NumPy array를 PyTorch tensor로 변환
tensor_back = torch.from_numpy(numpy_array_back)
tensor_back = tensor_back.cuda()
print("Features of detected local object (back):", tensor_back)


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
        
        # Siamese Network를 통한 특성 추출 및 유사도 계산
        features_local_obj, features_roi = model_siamese(tensor_local_obj, tensor_roi)
        similarity = get_similarity(torch.tensor([[ 1.20046, -0.69305,  1.58131,  0.60816, -0.52821,  0.25740, -3.49271, -0.59694, -0.20467, -0.15504, -0.40855,  0.46700,  2.49057, -0.05614, -1.80193, -0.14389, -0.75003, -0.90647,  0.41928, -0.49404, -0.78048,  0.70899,  1.98358, -0.33704, -0.48695, -0.19231, -0.19496, -0.23030,  2.74465, -0.23027,  1.19443,
         -0.21312,  1.29959, -0.60199,  0.25927, -0.32549,  0.89090, -0.54627, -0.40614, -0.52519, -2.20817, -0.20998,  2.25398,  0.78721,  1.69795,  0.53083, -0.76171,  0.06504, -0.43501, -0.39037,  0.77873,  0.15639,  1.73682, -1.48509,  0.60988, -1.00065, -0.42932, -0.83907,  0.88715, -0.11609,  1.35307, -0.53791,
          1.39896,  2.17149, -2.44928,  0.08313,  0.22740, -1.54707, -0.38557,  0.21895, -0.41478,  0.85844, -0.95989,  1.23016, -0.07023, -2.33856, -0.13861, -1.08122, -0.49841, -0.32174,  0.65510, -1.62068, -0.76745, -0.44805,  1.64729,  1.06194,  0.06727, -0.60062, -0.09979, -1.14660,  0.93203, -0.92535,  1.61461,
         -2.81766,  0.54675,  0.16358, -0.06079, -1.45224,  0.19950, -0.90520, -2.11403,  2.48722,  0.05627, -0.18189,  0.14696,  1.90275,  0.75048, -1.19118,  0.63959,  0.97017, -0.57070,  0.93061,  0.13470,  1.22751,  0.64398,  2.62456,  1.24027, -0.56505, -0.03781, -0.25886, -0.79504, -2.13959,  0.25158,  1.82397,
         -0.93647, -2.45491, -0.80061,  0.79180, -1.46602,  0.04989, -0.78122, -0.18620, -2.21684, -0.07015,  1.66183,  1.68905, -2.30047, -0.47894,  0.19492, -0.37816,  0.40222, -0.19167,  0.58801,  0.78380,  0.84957,  0.20700,  0.56888,  0.11780,  0.21148, -0.45224, -0.56288,  0.70653, -0.98885, -0.49305, -0.63546,
          0.20547, -0.20612,  0.14187, -3.24136,  0.12076,  1.10366, -0.62251,  0.18386,  0.37831,  0.72453, -1.04384, -0.38892,  0.00989,  0.00637,  1.76110,  2.86995,  0.79860,  1.20390, -1.10420,  1.01581, -0.98286, -0.85595, -1.12551, -0.14263,  1.04851,  1.58130,  0.78703, -1.00261,  0.69915,  1.33733,  0.04468,
         -2.47783, -0.08741,  0.35031, -2.18097,  0.55997, -0.17326, -0.29432, -2.60956,  1.98458, -0.95368, -0.83826, -0.55345,  1.03447,  0.26613, -0.47419,  1.28100,  0.66493, -0.05191, -0.42563, -0.97462,  0.24656,  0.50950, -1.63907, -2.85245, -0.04374, -1.40833,  0.07156,  1.42470, -1.95658, -0.45858, -2.14192,
         -0.90979, -0.96950, -1.73049,  1.08061,  1.74424, -1.11605, -1.33539,  0.58781, -1.33016,  0.69865, -0.20449,  1.13003, -0.47576,  1.06199,  0.04767,  1.00429, -0.57695, -1.58760, -1.10884, -0.66922,  1.81693,  0.49854,  1.74981,  2.70757,  0.44286,  1.69010,  2.74095,  1.12663,  0.67640, -0.66431,  1.21351,
          0.99942, -0.23385,  0.69010, -0.90816,  1.21524,  1.49552,  1.32817, -0.43630]], device='cuda:0'), features_roi)
        
        label = f"Sim: {similarity.item():.2f}"
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