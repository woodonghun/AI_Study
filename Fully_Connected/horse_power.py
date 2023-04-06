import time

import pandas as pd  # 데이터프레임 형태를 다룰 수 있는 라이브러리
import numpy as np

# ANN
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim  # torch 내의 세부적인 기능을 불러온다. (신경망 기술, 손실함수, 최적화 방법 등)
from torch.utils.data import DataLoader, Dataset, random_split  # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리
import torch.nn.functional as F  # torch 내의 세부적인 기능을 불러온다. (신경망 기술 등)

# Loss
from sklearn.metrics import mean_squared_error  # Regression 문제의 평가를 위해 MSE(Mean Squared Error)를 불러온다.

from torch.utils.tensorboard import SummaryWriter

'''
    tensorboard 사용시 Terminal 에서 tensorboard --logdir=./~~~
    logdir => log_dir 에서 작성한 경로의 상위 폴더
'''
learning_rate = 0.0001
batch_size = 15
epoch_size = 400
weight_decay = 1e-7
project_name = 'horsepower'

model_save_path = r'D:\AI 공부\3. 자동차 마력\model'

pretrained_model = 0  # 0 일 때는 사전 학습 없음, 1일때 사전 학습 있음
model_name = 'temp.pt'

''''''''''''''''''''''''''''''''''''''

# 임시로 지정
tensorboard_file_name = f"{time.strftime('%H%M%S')}_epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}"

writer = SummaryWriter(log_dir=f'{project_name}/{tensorboard_file_name}', filename_suffix=tensorboard_file_name,
                       comment=f"epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}")

columns = ['mpg', 'cylinders', ' displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']


class Tensordata(Dataset):
    def __init__(self, file_path):
        scaler_minmax = MinMaxScaler()

        df = pd.read_csv(file_path, names=columns)
        df = df.replace('?', None)  # ? => 결측치 변경
        df = df.dropna()  # 결측치 있는 행 제거

        origin = df.pop('origin')  # origin 열 빼옴

        # origin: 제조 장소(1: 미국 USA, 2: 유럽 EU, 3: 일본 JPN)
        # one hot encoding -
        df['USA'] = (origin == 1) * 1
        df['EU'] = (origin == 2) * 1
        df['JPN'] = (origin == 3) * 1

        print(df)

        self.x = df.drop(['car name'], axis=1).to_numpy()
        scaler_minmax.fit(self.x)
        self.x = scaler_minmax.transform(self.x)
        self.x = torch.FloatTensor(self.x)

        self.y = df['mpg'].to_numpy().reshape((-1, 1))  # 타겟
        scaler_minmax.fit(self.y)
        self.y = scaler_minmax.transform(self.y)
        self.y = torch.FloatTensor(self.y)

        self.len = len(df)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


dataset = Tensordata(r'D:\AI 공부\3. 자동차 마력\auto-mpg.csv')
data_size = len(dataset)

train_size = int(data_size * 0.8)
vaild_size = int(data_size * 0.1)
test_size = data_size - train_size - vaild_size

train_dataset, vaild_dataset, test_dataset = random_split(dataset, [train_size, vaild_size, test_size])  # Train Test Vaild 분리

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
vaildloader = DataLoader(vaild_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

print(f'total Data Size : {data_size}')
print(f"Training Data Size : {len(train_dataset)}")
print(f"Validation Data Size : {len(vaild_dataset)}")
print(f"Testing Data Size : {len(test_dataset)}")

print(len(trainloader), len(vaildloader), len(testloader))


class Regressor(nn.Module):  # 모델
    def __init__(self):
        super().__init__()  # 모델 연산 정의
        self.fc1 = nn.Linear(10, 45, bias=True)
        self.fc2 = nn.Linear(45, 30, bias=True)
        self.fc3 = nn.Linear(30, 1, bias=True)
        self.dropout = nn.Dropout(0.2)  # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.

    def forward(self, x):  # 모델 연산의 순서를 정의
        x = F.relu(self.fc1(x))  # Linear 계산 후 활성화 함수 ReLU를 적용한다.
        x = self.dropout(F.relu(self.fc2(x)))  # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        x = F.relu(self.fc3(x))  # Linear 계산 후 활성화 함수 ReLU를 적용한다.

        return x


# lr은 학습률이다.
# weight_decay 는 L2 정규화에서의 penalty 정도를 의미한다.
model = Regressor()
criterion = nn.MSELoss()
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

    running_train_loss = 0.0
    train_rmse = 0.0
    vaild_rmse = 0.0
    running_vaild_loss = 0.0

    # train
    for i, data in enumerate(trainloader, 0):
        inputs, values = data  # data에는 X, Y가 들어있다.

        optimizer.zero_grad()  # 최적화 초기화

        predict_outputs = model(inputs)  # 모델에 입력값 대입 후 예측값 산출
        # print(values, predict_outputs)
        loss = criterion(predict_outputs, values)  # 손실 함수 계산
        loss.backward()  # 손실 함수 기준으로 역전파 설정
        optimizer.step()  # 역전파를 진행하고 가중치 업데이트 / 최적화
        running_train_loss += loss.item()  # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.
        predict_outputs = predict_outputs.detach()
        train_rmse = np.sqrt(mean_squared_error(predict_outputs, values))  # sklearn을 이용하여 RMSE 계산

    # vaild
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(vaildloader, 0):
            inputs, values = data

            vaild_output = model(inputs)
            loss = criterion(vaild_output, values)
            running_vaild_loss += loss.item()
            vaild_rmse = np.sqrt(mean_squared_error(vaild_output, values))  # sklearn을 이용하여 RMSE 계산

    # 모델 저장
    loss_save = running_vaild_loss / len(vaildloader)
    loss_.append(loss_save)
    # loss 값이 작아질 때 마다 저장
    if loss_save < ls:
        torch.save({'epoch': epoch,
                    'loss': loss_,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, fr'{model_save_path}\{model_name}')

    # Tensorboard 에 저장
    writer.add_scalar("train/rmse", train_rmse, epoch)
    writer.add_scalar("train/loss", running_train_loss / len(trainloader), epoch)
    writer.add_scalar('vaild/rmse', vaild_rmse, epoch)
    writer.add_scalar('vaild/loss', running_vaild_loss / len(vaildloader), epoch)
    writer.add_scalars('loss', {'train': running_train_loss / len(trainloader), 'vaild': running_vaild_loss / len(vaildloader)}, epoch)
    writer.add_scalars('rmse', {'train': train_rmse, 'vaild': vaild_rmse}, epoch)

    # epoch 당 loss rmse 출력
    if epoch % 10 == 9:
        print(
            f"epoch {epoch + 1} -"
            f" train loss: {running_train_loss / len(trainloader):.8f},"
            f" train mse : {train_rmse:.8f}"
            f" ----------"
            f" vaild loss : {running_vaild_loss / len(vaildloader):.8f}"
            f" vaild mse : {vaild_rmse:.8f}"
        )

writer.flush()
writer.close()

train_rmse = 0.0
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, values = data
        test_output = model(inputs)
        print(f'예측 {test_output.view(1, -1)} \n실제 값 {values.view(1, -1)}')
        train_rmse = np.sqrt(mean_squared_error(test_output, values))  # sklearn을 이용하여 RMSE 계산
        print(train_rmse)
