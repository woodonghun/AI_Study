import time
import seaborn as sns

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
learning_rate = 0.0001
batch_size = 32
epoch_size = 2000
weight_decay = 1e-3
project_name = 'wine_quality'

model_save_path = r'D:\AI 공부\4. 와인 퀄리티\model'

pretrained_model = 0  # 0 일 때는 사전 학습 없음, 1일때 사전 학습 있음
model_name = 'temp.pt'

''''''''''''''''''''''''''''''''''''''

tensorboard_file_name = f"{time.strftime('%H%M%S')}_epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}"
writer = SummaryWriter(log_dir=f'{project_name}/{tensorboard_file_name}', filename_suffix=tensorboard_file_name,
                       comment=f"epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}")


# iqr 이상치 제거 함수
def detect_outliers(df, columns):
    q1 = df[columns].quantile(0.25)
    q3 = df[columns].quantile(0.75)
    iqr = q3 - q1

    boundary = 1.5 * iqr

    index1 = df[df[columns] > q3 + boundary].index
    index2 = df[df[columns] < q1 - boundary].index

    df = df.drop(index1)
    df = df.drop(index2)

    return df


class CustomDataset(Dataset):
    def __init__(self, file_path):
        scaler_minmax = MinMaxScaler()

        dataframe = pd.read_csv(file_path)
        sns.boxplot(data=dataframe)
        # plt.show()
        print(dataframe)
        # 이상치 제거
        dataframe = detect_outliers(dataframe, 'fixed acidity')
        dataframe = detect_outliers(dataframe, 'volatile acidity')
        dataframe = detect_outliers(dataframe, 'citric acid')
        dataframe = detect_outliers(dataframe, 'residual sugar')
        dataframe = detect_outliers(dataframe, 'chlorides')
        dataframe = detect_outliers(dataframe, 'free sulfur dioxide')
        dataframe = detect_outliers(dataframe, 'total sulfur dioxide')
        dataframe = detect_outliers(dataframe, 'density')
        dataframe = detect_outliers(dataframe, 'pH')
        dataframe = detect_outliers(dataframe, 'sulphates')
        dataframe = detect_outliers(dataframe, 'alcohol')
        sns.boxplot(data=dataframe)
        # plt.show()

        print(dataframe)

        self.y = dataframe['quality'].to_numpy()  # target
        # scaler_minmax.fit(self.y)
        # self.y = scaler_minmax.transform(self.y)
        self.y = self.y - 3
        self.y = torch.LongTensor(self.y)

        self.x = dataframe.drop(['quality'], axis=1)  # data
        scaler_minmax.fit(self.x)
        self.x = scaler_minmax.transform(self.x)
        self.x = torch.FloatTensor(self.x)
        print(self.x)
        self.len = len(dataframe)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


dataset = CustomDataset(r'D:\AI 공부\4. 와인 퀄리티\winequality-red.csv')  # 커스텀데이터셋 생성
data_size = len(dataset)

train_size = int(data_size * 0.7)
vaild_size = int(data_size * 0.2)
test_size = data_size - train_size - vaild_size

train_dataset, vaild_dataset, test_dataset = random_split(dataset, [train_size, vaild_size, test_size])  # Train Test Vaild 분리

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
vaildloader = DataLoader(vaild_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

print(f'total Data Size : {data_size}')
print(f"Training Data Size : {len(train_dataset)}")
print(f"Validation Data Size : {len(vaild_dataset)}")
print(f"Testing Data Size : {len(test_dataset)}")


class Model(nn.Module):  # 모델
    def __init__(self):
        super().__init__()  # 모델 연산 정의
        self.fc1 = nn.Linear(11, 16)
        self.fc2 = nn.Linear(16, 32)
        # self.fc3 = nn.Linear(32, 64)
        # self.fc4 = nn.Linear(32, 128)
        # self.fc5 = nn.Linear(32, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 6)

        self.dropout = nn.Dropout(0.2)  # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.

    def forward(self, x):  # 모델 연산의 순서를 정의
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


model = Model()
criterion = nn.CrossEntropyLoss()
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
    valid_acc = 0.0
    valid_loss = 0.0

    # train
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, values = data  # data에는 X, Y가 들어있다.
        optimizer.zero_grad()  # 최적화 초기화

        train_outputs = model(inputs)  # 모델에 입력값 대입 후 예측값 산출

        # print(train_outputs,values)
        loss = criterion(train_outputs, values)  # 손실 함수 계산
        loss.backward()  # 손실 함수 기준으로 역전파 설정
        optimizer.step()  # 역전파를 진행하고 가중치 업데이트 / 최적화

        train_loss += loss.item()  # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.
        _, outputs = torch.max(train_outputs, 1)  # 가장 큰 값을 가져와서 value 값과 비교 할수 있다.
        train_acc += (outputs == values).sum()

        # print(train_output)
    # vaild
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(vaildloader, 0):
            inputs, values = data

            valid_outputs = model(inputs)
            loss = criterion(valid_outputs, values)
            valid_loss += loss.item()
            _, outputs = torch.max(valid_outputs, 1)
            # print(outputs,values)
            valid_acc += (outputs == values).sum()
    # 모델 저장
    loss_save = valid_loss / len(vaildloader)
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
    writer.add_scalar('vaild/acc', 100 * valid_acc / len(vaild_dataset), epoch)
    writer.add_scalar('vaild/loss', valid_loss / len(vaildloader), epoch)
    writer.add_scalars('loss', {'train': train_loss / len(trainloader), 'vaild': valid_loss / len(vaildloader)}, epoch)
    writer.add_scalars('acc', {'train': 100 * train_acc / len(train_dataset), 'vaild': 100 * valid_acc / len(vaild_dataset)}, epoch)

    # epoch 당 loss rmse 출력
    # if epoch % 10 == 9:
    print(
        f"epoch {epoch + 1} -"
        f" train loss: {train_loss / len(trainloader):.8f},"
        f" train acc : {100 * train_acc / len(train_dataset):.8f}"
        f" ----------"
        f" vaild loss : {valid_loss / len(vaildloader):.8f}"
        f" vaild acc : {100 * valid_acc / len(vaild_dataset):.8f}"
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
        _, outputs = torch.max(test_output, 1)

        acc += (outputs == values).sum()

        print(f'예측 {outputs.tolist()} \n실제 값 {values.tolist()}')
        # acc += (values == test_output).sum()
    print(100 * acc / len(test_dataset))
