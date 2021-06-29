"""
화장 안 한 이미지에 화장 한 이미지를 학습시키기
"""

import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf  #뷰티 gan은 텐서버전1에서 만들어서 1로 변환해준다
tf.disable_v2_behavior()
import numpy as np

detector = dlib.get_frontal_face_detector()  #이미지에서 얼굴만 찾아주는 함수

# 얼굴의 랜드마크 점의 위치를 찾아주는 모델을 받아 불러오기
sp = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

#이미지 한 장 불러오기(확인용)
img = dlib.load_rgb_image('./imgs/NA.jpg')
plt.figure(figsize=(16, 10))   #이미지 사이즈 지정
plt.imshow(img)
plt.show()

#이미지에서 사람의 얼굴에 사각형 그리기
img_result = img.copy()  #원본사진 복사하기
dets = detector(img)  #이미지에서 얼굴의 좌표를 찾아주기(얼굴이 여러개면 각각의 얼굴을 다 찾아준다)
if len(dets) == 0:
    print('cannot find faces!')
else:
    fig, ax = plt.subplots(1, figsize=(16, 10))  #이미지 사이즈 지정
    for det in dets:  #얼굴이 여러개일 수 있으니 for문을 사용한다
        x, y, w, h = det.left(), det.top(), det.width(), det.height()    #얼굴의 좌표의 사이즈인 왼쪽, 위쪽, 폭, 높이의 데이터를 받는 것
        # 위 데이터로 사각형을 만들어준다, linewidth=선의 두께, 색은 빨간색, facecolor=사각형안에 색깔을 지정
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()


#얼굴 이목구비에 원으로 점 그리기
fig, ax = plt.subplots(1, figsize=(16, 10))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r')
        ax.add_patch(circle)

ax.imshow(img_result)
plt.show()

#얼굴만 잘라서 가져오기
faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)  #padding을 하면 얼굴 주면에 패딩이 된다(얼굴이 너무 잘릴까봐)
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20, 16))
axes[0].imshow(img)  #원본 사진을 제일 앞에 출력
for i, face in enumerate(faces):  #얼굴마다 따로 보여준다
    axes[i+1].imshow(face)
plt.show()
