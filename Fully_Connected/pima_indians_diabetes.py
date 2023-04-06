import time

import pandas as pd  # 데이터프레임 형태를 다룰 수 있는 라이브러리
import numpy as np
from sklearn.model_selection import train_test_split  # 전체 데이터를 학습 데이터와 평가 데이터로 나눈다.

# ANN
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim  # torch 내의 세부적인 기능을 불러온다. (신경망 기술, 손실함수, 최적화 방법 등)
from torch.utils.data import DataLoader, Dataset, random_split  # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리
import torch.nn.functional as F  # torch 내의 세부적인 기능을 불러온다. (신경망 기술 등)

# Loss
from sklearn.metrics import mean_squared_error  # Regression 문제의 평가를 위해 MSE(Mean Squared Error)를 불러온다.

# Plot
import matplotlib.pyplot as plt  # 시각화 도구

from torch.utils.tensorboard import SummaryWriter

'''
    tensorboard 사용시 Terminal 에서 tensorboard --logdir=./~~~
    logdir => log_dir 에서 작성한 경로의 상위 폴더
'''
learning_rate = 0.01
batch_size = 16
epoch_size = 500
weight_decay = 1e-7
project_name = 'pima'

data_path = r'D:\AI 공부\8. 피마 인디언 당뇨병\diabetes.csv'
model_save_path = r'D:\AI 공부\8. 피마 인디언 당뇨병\model'

pretrained_model = 0  # 0 일 때는 사전 학습 없음, 1일때 사전 학습 있음
model_name = 'temp.pt'

''''''''''''''''''''''''''''''''''''''

tensorboard_file_name = f"{time.strftime('%H%M%S')}_epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}"
writer = SummaryWriter(log_dir=f'{project_name}/{tensorboard_file_name}', filename_suffix=tensorboard_file_name,
                       comment=f"epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}")


class Tensordata(Dataset):
    def __init__(self, file_path):
        scaler_minmax = MinMaxScaler()

        dataframe = pd.read_csv(file_path)
        print(dataframe.shape)
        print(dataframe.count())
        self.y = dataframe['Outcome'].to_numpy().reshape((-1, 1))  # target
        # scaler_minmax.fit(self.y)
        # self.y = scaler_minmax.transform(self.y)
        self.y = torch.FloatTensor(self.y)

        self.x = dataframe.drop(['Outcome'], axis=1)  # data
        scaler_minmax.fit(self.x)
        self.x = scaler_minmax.transform(self.x)
        self.x = torch.FloatTensor(self.x)
        self.len = len(dataframe)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


dataset = Tensordata(data_path)
data_size = len(dataset)

train_size = int(data_size * 0.8)
vaild_size = int(data_size * 0.1)
test_size = data_size - train_size - vaild_size

train_dataset, vaild_dataset, test_dataset = random_split(dataset, [train_size, vaild_size, test_size])  # Train Test Vaild 분리

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
vaildloader = DataLoader(vaild_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

print(f'Dataset shape - train : {train_dataset} vaild : {vaild_dataset} test : {test_dataset}')

print(f'total Data Size : {data_size}')
print(f"Training Data Size : {len(train_dataset)}")
print(f"Validation Data Size : {len(vaild_dataset)}")
print(f"Testing Data Size : {len(test_dataset)}")

print(len(trainloader), len(vaildloader), len(testloader))


class Model(nn.Module):  # 모델
    def __init__(self):
        super().__init__()  # 모델 연산 정의
        self.fc1 = nn.Linear(8, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, 1, bias=True)
        self.dropout = nn.Dropout(0.4)  # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.

    def forward(self, x):  # 모델 연산의 순서를 정의
        x = F.relu(self.fc1(x))  # Linear 계산 후 활성화 함수 ReLU를 적용한다.
        x = self.dropout(F.relu(self.fc2(x)))  # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        x = torch.sigmoid(self.fc3(x))  # Linear 계산 후 활성화 함수 ReLU를 적용한다.

        return x


model = Model()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

if pretrained_model == 1:
    checkpoint = torch.load(fr'{model_save_path}\{model_name}')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_ = checkpoint['loss']
    ep = checkpoint['epoch']
    ls = loss_[-1]  # 제일 마지막 로스값
    print(f"epoch={ep}, loss={ls}")
    ep = ep + 1
else:
    ep = 0
    ls = 1

loss_ = []  # 그래프를 그리기 위한 loss 저장용 리스트
n = len(trainloader)

for epoch in range(epoch_size):

    train_loss = 0.0
    train_acc = 0.0
    vaild_loss = 0.0
    vaild_acc = 0.0

    # train
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, values = data  # data에는 X, Y가 들어있다.

        optimizer.zero_grad()  # 최적화 초기화

        train_outputs = model(inputs)  # 모델에 입력값 대입 후 예측값 산출
        print(values, train_outputs)
        loss = criterion(train_outputs, values)  # 손실 함수 계산
        loss.backward()  # 손실 함수 기준으로 역전파 설정
        optimizer.step()  # 역전파를 진행하고 가중치 업데이트 / 최적화
        train_loss += loss.item()  # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.

        test_output = torch.round(train_outputs)  # 반올림
        train_acc += (values == test_output).sum()

    # vaild
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(vaildloader, 0):
            inputs, values = data

            vaild_output = model(inputs)
            loss = criterion(vaild_output, values)
            vaild_loss += loss.item()
            vaild_output = torch.round(vaild_output)  # 반올림
            vaild_acc += (values == vaild_output).sum()

    # 모델 저장
    loss_save = vaild_loss / len(vaildloader)
    loss_.append(loss_save)

    # loss 값이 작아질 때 마다 저장
    if loss_save < ls:
        torch.save({'epoch': epoch,
                    'loss': loss_,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, fr'{model_save_path}\{model_name}')

    # Tensorboard 에 저장
    writer.add_scalar("train/acc", 100 * train_acc / len(train_dataset), epoch)
    writer.add_scalar("train/loss", train_loss / len(trainloader), epoch)
    writer.add_scalar('vaild/acc', 100 * vaild_acc / len(vaild_dataset), epoch)
    writer.add_scalar('vaild/loss', vaild_loss / len(vaildloader), epoch)
    writer.add_scalars('loss', {'train': train_loss / len(trainloader), 'vaild': vaild_loss / len(vaildloader)}, epoch)
    writer.add_scalars('acc', {'train': 100 * train_acc / len(train_dataset), 'vaild': 100 * vaild_acc / len(vaild_dataset)}, epoch)

    # epoch 당 loss rmse 출력
    if epoch % 50 == 49:
        print(
            f"epoch {epoch + 1} -"
            f" train loss: {train_loss / len(trainloader):.8f},"
            f" train mse : {100 * train_acc / len(train_dataset):.8f}"
            f" ----------"
            f" vaild loss : {vaild_loss / len(vaildloader):.8f}"
            f" vaild mse : {100 * vaild_acc / len(vaild_dataset):.8f}"
        )

writer.flush()
writer.close()

acc = 0.0

checkpoint = torch.load(f'{model_save_path}\{model_name}')
model.load_state_dict(checkpoint['model'])
print(model)
model.eval()
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, values = data
        test_output = model(inputs)
        # print(test_output.tolist())
        test_output = torch.round(test_output)
        print(f'예측 {test_output.tolist()} \n실제 값 {values.tolist()}')
        acc += (values == test_output).sum()
    print(100 * acc / len(test_dataset))
