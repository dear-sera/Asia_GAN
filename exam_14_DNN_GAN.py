"""mnist로 GAN 실습해보기"""

#모듈 불러오기
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

#이미지가 들어간 경로 지정하기
OUT_DIR = './OUT_img/'

#지정된 수치는 미리 변수로 지정해주기
img_shape = (28, 28, 1)
epoch = 100000
batch_size = 128
noise = 100
simple_interval = 100

#학습에 들어갈 이미지 데이터 불러 학습, 검증데이터로 나누기
(X_train, _), (_, _) = mnist.load_data()
print(X_train.shape)  #결과 (60000, 28, 28)

#데이터 스케일링하기
X_train = X_train / 127.5 - 1  #255로 나누지 않고 이렇게 나누는 이유는 x_train이 127.5로 나눴을 때 2가 되서, tanh를 활용하면 output이 -1 ~ 1 사이로 나오기 때문에

#expand_dims = reshape은 같은 원리
X_train = np.expand_dims(X_train, axis=3)  #axis=3을 줘서 차원늘리기
print(X_train.shape)

#build generator = 100개짜리 잡음이 들어가는 모델 생성하기
generator_model = Sequential()
generator_model.add(Dense(128, input_dim=noise))  #랜덤으로 들어갈 잡음의 차원의 수 100개
generator_model.add(LeakyReLU(alpha=0.01))  #LeakyReLU=activation함수라서 여기까진 레이어가 1층으로 이루어짐, LeakyReLU=relu와 같지만 마이너스 값을 조금 학습할 수 있는 함수
generator_model.add(Dense(784, activation='tanh'))  #784픽셀의 이미지가 1장이 나온다
generator_model.add(Reshape(img_shape))  #출력을 이미지로 바꿔주기 위해 reshape
print(generator_model.summary())

#build discriminator = 진품 데이터(mnist)를 넣어줄 모델 생성하기
lrelu = LeakyReLU(alpha=0.01)  #LeakyReLU는 alpha 값을 줘야하기 때문에 한 줄에 못적는다 그래서 변수로 지정해서 넣기
discriminator_model = Sequential()
discriminator_model.add(Flatten(input_shape=img_shape))  #모델에서 flatten을 사용하면 한 줄로 reshape해줘서 사용 (들어간 사진을 확인하려면 형태를 유지해야해서), 파라미터개수o
discriminator_model.add(Dense(128, activation=lrelu))
#discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Dense(1, activation='sigmoid'))  #이진분류기(진품인지 가품인지 확인하기 위해서)라서 1개
print(discriminator_model.summary())


discriminator_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
discriminator_model.trainable = False  #False로 설정하면 학습을 하지 않는다

#build GAN
gan_model = Sequential()
gan_model.add(generator_model)  #이렇게 모델에 기존 모델을 이어붙이려면 add로 넣으면 이어진다
gan_model.add(discriminator_model)
print(gan_model.summary())

#gan model compile
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

#진품에서 쓸 정답 만들기
real = np.ones((batch_size, 1))  #ones는 모든 값을 1로 만들어준다, 1개씩 든 리스트 128개로 나온다(배치사이즈128)
print(real)

fake = np.zeros((batch_size, 1))  #zeros= 모든 값을 0으로 만들어준다, 1개씩 든 리스트 128개로 나온다(배치사이즈128)
print(fake)

for itr in range(epoch):  #십만번을 학습 시켜서 mnist에는 없지만 비슷한 이미지를 만들어준다
    # X_train의 shape는 (60000, 28, 28)이라서 [0]는 60000 = 0~59999 사이의 int값을 배치사이즈 개수만큼 랜덤하게 추출
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]  #128개가 인덱싱 된다

    # 잡음 만들기(batch_size, noise) = (128, 100) = 랜덤하게 만들어진 100개데이터를 128개를 만들어준다, 이 데이터의 평균은 0 표준편차 1인 데이터 만들기
    z = np.random.normal(0, 1, (batch_size, noise))
    fake_imgs = generator_model.predict(z)  #784픽셀짜리 이미지 128장

    #정답은 1, 오답은 0으로 나오게 학습시키기
    d_hist_real = discriminator_model.train_on_batch(real_imgs, real)  #정답이미지와 정답을 주고 학습시키기
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake)  #generator가 만들어낸 fake이미지를 주고 0을 준 뒤 학습시키기

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)  #두 가지를 합쳐 평균낸 것
    discriminator_model.trainable = False  #gan_model학습 시엔 generator만 학습하기위해 false를 준다, 위에서 학습했는데 gan에서 학습을 한 번 더 하지 않게 하기 위해

    z = np.random.normal(0, 1, (batch_size, noise))
    gan_hist = gan_model.train_on_batch(z, real)  #가짜지만 real로 라벨을 달아준다, 간모델은 출력이 1이 나오게 해야해서 준 것

    #100 epoch마다 이미지를 그려서 저장하기
    if itr % simple_interval == 0:
        print('%d [D ;pss: %f, acc.: %.2f%%] [G loss: %f]'%(itr, d_loss, d_acc*100, gan_hist))  #d_acc*100=%나오게 100을 곱해준다
        row = col = 4  #4*4=16개의 이미지 내기
        z = np.random.normal(0, 1, (row*col, noise))
        fake_imgs = generator_model.predict((z))
        fake_imgs = 0.5 * fake_imgs + 0.5 #스케일로 마이너스 값이 되서, 다시 0이상으로 만들기
        _, axs = plt.subplots(row, col, figsize=(row, col), sharey=True, sharex=True)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cnt, :, 0], cmap='gray')
                axs[i, j].axis('off')  #axis('off') = x, y축의 눈금 지우기
                cnt += 1

        #plot 저장하기
        path = os.path.join(OUT_DIR, f"img-{itr + 1}")
        plt.savefig(path)
        plt.close()  #코드 실행 시 plot이 메모리에 쌓이지 않게 닫아주는 것

