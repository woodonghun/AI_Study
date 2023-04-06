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
learning_rate = 0.001
batch_size = 1024
epoch_size = 60
weight_decay = 1e-7
project_name = 'fashion_mnist'

data_path_train = r'D:\AI 공부\ann\6. fashion mnist\fashion-mnist_train.csv'
data_path_test = r'D:\AI 공부\ann\6. fashion mnist\fashion-mnist_test.csv'
model_save_path = r'D:\AI 공부\ann\6. fashion mnist\model'

pretrained_model = 0  # 0 일 때는 사전 학습 없음, 1일때 사전 학습 있음
model_name = 'temp.pt'

''''''''''''''''''''''''''''''''''''''

tensorboard_file_name = f"{time.strftime('%H%M%S')}_epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}"
writer = SummaryWriter(log_dir=f'{project_name}/{tensorboard_file_name}', filename_suffix=tensorboard_file_name,
                       comment=f"epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}")


class Tensordata(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.y = self.data['label'].to_numpy()  # target
        self.x = self.data.drop(['label'], axis=1)  # data
        self.len = len(self.data)

        scaler_minmax = MinMaxScaler()
        self.y = torch.LongTensor(self.y)
        print(self.y)
        scaler_minmax.fit(self.x)
        self.x = scaler_minmax.transform(self.x)
        self.x = torch.FloatTensor(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


dataset_train = Tensordata(data_path_train)
dataset_test = Tensordata(data_path_test)
data_size = len(dataset_train)

train_size = int(data_size * 0.8)
vaild_size = data_size - train_size

train_dataset, vaild_dataset = random_split(dataset_train, [train_size, vaild_size])  # Train Test Vaild 분리

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
vaildloader = DataLoader(vaild_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, drop_last=False)

print(f"Training Data Size : {len(train_dataset)}")
print(f"Validation Data Size : {len(vaild_dataset)}")
print(f"Testing Data Size : {len(dataset_test)}")

print(len(trainloader), len(vaildloader), len(testloader))

fashion = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bang', 9: 'Ankle boot'}


class Model(nn.Module):  # 모델
    def __init__(self):
        super().__init__()  # 모델 연산 정의
        self.fc1 = nn.Linear(784, 500, bias=True)
        self.fc2 = nn.Linear(500, 30, bias=True)
        self.fc3 = nn.Linear(30, 10, bias=True)

        self.dropout = nn.Dropout(0.4)  # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.

    def forward(self, x):  # 모델 연산의 순서를 정의
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))  # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        x = F.relu(self.fc3(x))

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

loss_ = []  # loss 값 저장용

for epoch in range(epoch_size):

    train_loss = 0.0
    train_acc = 0.0
    vaild_loss = 0.0
    vaild_acc = 0.0

    # train
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, values = data  # data에는 X, Y가 들어있다.
        inputs = inputs.view(-1, 784)

        optimizer.zero_grad()  # 최적화 초기화

        train_outputs = model(inputs)  # 모델에 입력값 대입 후 예측값 산출
        print(train_outputs, values)
        loss = criterion(train_outputs, values)  # 손실 함수 계산
        loss.backward()  # 손실 함수 기준으로 역전파 설정
        optimizer.step()  # 역전파를 진행하고 가중치 업데이트 / 최적화
        train_loss += loss.item()  # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.
        _, outputs = torch.max(train_outputs, 1)  # 가장 큰 값을 가져와서 value 값과 비교 할수 있다.
        train_acc += (outputs == values).sum()
        # print(f'train: {train_outputs.tolist()}\n, value: {values.tolist()}')

    # vaild
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(vaildloader, 0):
            inputs, values = data

            vaild_output = model(inputs)
            loss = criterion(vaild_output, values)
            vaild_loss += loss.item()
            _, outputs = torch.max(vaild_output, 1)
            # print(outputs,values)
            vaild_acc += (outputs == values).sum()

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
    # if epoch % 10 == 9:
    print(
        f"epoch {epoch + 1} -"
        f" train loss: {train_loss / len(trainloader):.8f},"
        f" train acc : {100 * train_acc / len(train_dataset):.8f}"
        f" ----------"
        f" vaild loss : {vaild_loss / len(vaildloader):.8f}"
        f" vaild acc : {100 * vaild_acc / len(vaild_dataset):.8f}"
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
    print(100 * acc / len(dataset_test))
