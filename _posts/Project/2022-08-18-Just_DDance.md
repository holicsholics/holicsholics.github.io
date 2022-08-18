---
title: Just DDance - 모션 인식을 통한 안무 연습 서비스 개발기
date: 2022-08-18T07:55:52.618Z

categories:
  - Project
tags:
  - Deep Learning
  - Machine Learning
  - mediapipe
  - numpy
  - pandas
  - matplot
---
# Just DDance 개발기
## 주제 선정 및 배경
코로나의 영향으로 실내 활동의 중요성이 대두 된 상황에서, 실내에서 즐길 수 있는 여가 활동의 다양성 확보를 위한 일환 중 하나로 선정하게 되었다.  

춤을 연습하는 경우, 일반적으로 안무 영상을 보며 따라하는 방식으로 연습을 하는데, 단순히 영상을 보며 따라하는 방법이 아닌, 화면에 가이드 라인이 표시되고 실시간으로 피드백을 주는 서비스를 기획했다.  
기존에 있는 "Just Dance"라는 게임을 보면, 모든 안무를 보여주는 것이 아니라 일부만 보여주며 해당 부분에 맞춰 춤을 춰야한다. 포즈 디텍션을 이용하면, 춤의 모든 부분을 보여주고 따라하면서 연습 할 수 있지 않을까라는 접근이었다.

## 개발기
기본적으로 각 기능에 대해서 모듈로 테스트를 해보고, 클래스 형식으로 합쳤다.  
의도했던 모든 기능이 구현되면 py 형태로 바꿔 어플리케이션을 제작해 볼 생각이었는데 한계가 있었다.
### 사용 라이브러리
```python
import os
import numpy as np
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import pytube
from ffpyplayer.player import MediaPlayer
```

### 데이터 수집
해당 프로젝트에서 사용한 데이터는 유튜브에서 다운 받은 안무 영상이다.  
기본적으로 한 명이 나오는 사람에 대한 영상을 사용했는데, `mediapipe`에서는 두 명 이상의 사람이 등장하면 객체 검출 성능이 떨어진다는 문제 때문이었다.  
`openpose`를 이용해 여러 사람에 대한 포즈 추정이나 `YOLO` 모델을 사용해서 처리하는 방법도 시도해봤지만, 이것들도 이것나름의 새로운 테스크들이 되서, 한 명의 사람만 등장한다는 가정으로 프로젝트를 진행했다.  
```python
def download_video(self):
    self.__save_dance_name()
    url = input(f"{self.__dance_name}의 안무 영상 링크: ")
    if not os.path.exists(self.__video_download_path): os.mkdir(self.__video_download_path)
    yt = pytube.YouTube(url).streams.filter(res="720p").first()
    yt.download(output_path=self.__video_download_path, filename=self.__dance_name+".mp4")
```

### 키포인트 추출
```python
def extract_keypoints(self, isMirr=False, showExtract=False):
        if not os.path.exists(self.__keypoints_path): os.mkdir(self.__keypoints_path)
        
        keypoint_dict_pose = []
        
        cv2.startWindowThread()
        cap = cv2.VideoCapture(os.path.join(self.__video_download_path, self.__dance_name+".mp4"))
        with self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret: break
                if not isMirr: image = cv2.flip(image, 1)
                
                results = pose.process(image)
                # Extracting
                try: keypoint_dict_pose.append({str(idx): [lmk.x, lmk.y, lmk.z] for idx, lmk in enumerate(results.pose_landmarks.landmark)})
                except: pass
                if showExtract:
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(244, 244, 244), thickness=2, circle_radius=1),
                                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(153, 255, 153), thickness=2, circle_radius=1))
                    cv2.imshow("Extracting", image)
                    if cv2.waitKey(1)==ord("q"): break
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        # Save coord. Data for json type
        with open(self.__keypoints_path+"/"+self.__dance_name+"_keypoints.json", "w") as keypoints:
            json.dump(keypoint_dict_pose, keypoints)
```
기본적인 프로세스는 영상에서 키포인트들을 추출하고 해당 키포인트를 저장해, 추후에 사용하는 방식이다.  
로직은 간단한데, 디텍션 모델을 활용해 영상의 프레임 단위로 키포인트를 추출해 저장한다.  

초기에는 `mediapipe`의 `holistic` 모델을 사용해서 포즈뿐만 아니라, FaceMesh와 손에 대한 키포인트도 추출했었는데, 얼굴이나 손은 포즈에 비해서 키포인트 추출이 잘 안되기도해서 프로토타입 제작시에는 사용하지 않았다.  
일반적인 영상의 길이가 3분 대이고, 해당 영상에 대해서 평균적으로 6000~7000 프레임의 포즈가 수집되는 반면, 얼굴이나 손은 1/5 수준으로만 수집되었다.  
수집되지 않는 원인은 디텍션이 잘되지 않아서인데, 이 부분은 직접 모델을 생성해서 해결할 수 있지만 또 다른 문제가 되어서 포즈만 사용했다.  

```python
keypoint_dict_pose.append(
    {str(idx): [lmk.x, lmk.y, lmk.z] for idx, lmk in enumerate(results.pose_landmarks.landmark)}
    )
```
Pose 키포인트의 경우 33개가 존재하고, 위와 같은 방식으로 각 프레임에 대해 모든 pose 키포인트를 수집해 json 형식으로 저장했다.

33개의 모든 키포인트가 매 프레임마다 수집되는건 아니므로 `try-except`로 감싸, 객체가 검출되지 않은 경우에도 다른 부분들은 수집할 수 있게 했다.

### 스케일링 및 출력
추출된 좌표들은 [x, y, z]로 3차원 좌표고, 해당 좌표들은 영상 사이즈에 맞게 0에서 1사이로 정규화 되어있다.  
예를 들어, 테스트에서 사용한 "[[주간아] 아이브 이서 러브다이브](https://www.youtube.com/watch?v=p6W1inGaUpo)"의 경우 408\*720 사이즈에 맞춰 정규화가 되어있다.  

사용자의 화면은 1280\*720으로 설정했는데, 해당 화면에 추출된 좌표를 알맞은 위치에 출력하기 위해서는 스케일링이 필요했다.

```python
try:
    # get coors MARGIN
    cors_margin = self.__get_margin([user_input["0"], user_input["23"], user_input["24"]], [dance_cors[dance_cors_frames][0], dance_cors[dance_cors_frames][23], dance_cors[dance_cors_frames][24]])
    for pose_point in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
        x_cor_pose, y_cor_pose, z_cor_pose = int((dance_cors[dance_cors_frames][pose_point][0]+cors_margin[0])*user_image.shape[1]), int((dance_cors[dance_cors_frames][pose_point][1]+cors_margin[1])*user_image.shape[0]), int((dance_cors[dance_cors_frames][pose_point][2]+cors_margin[2])*1000)
        cv2.circle(user_image, (x_cor_pose, y_cor_pose), 8, (244, 244, 244), cv2.FILLED)
        skeletons[pose_point] = (x_cor_pose, y_cor_pose)

        self.__draw_skeleton(user_image, skeletons)
        dance_cors_frames +=1
except: pass
```
![img](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)  
불러온 데이터는 행이 프레임 번호고, 각 열이 pose에 맵핑되어 있는 키이다.  
```python
dance_cors[0][0] # 첫번째 프레임의 코의 좌표
```
 
### 정확도 측정

## 테스트 영상

## 한계 및 개선 방안

## 전체 소스 코드