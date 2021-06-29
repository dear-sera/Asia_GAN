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
img = dlib.load_rgb_image('./imgs/12.jpg')
plt.figure(figsize=(16, 10))   #이미지 사이즈 지정
plt.imshow(img)
plt.show()


