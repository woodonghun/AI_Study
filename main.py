import PIL
import pandas as pd  # 데이터프레임 형태를 다룰 수 있는 라이브러리
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split  # 전체 데이터를 학습 데이터와 평가 데이터로 나눈다.

from PIL import Image

# ANN
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim  # torch 내의 세부적인 기능을 불러온다. (신경망 기술, 손실함수, 최적화 방법 등)
from torch.utils.data import DataLoader, Dataset  # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리
import torch.nn.functional as F  # torch 내의 세부적인 기능을 불러온다. (신경망 기술 등)

# Loss
from sklearn.metrics import mean_squared_error  # Regression 문제의 평가를 위해 MSE(Mean Squared Error)를 불러온다.

# Plot
import matplotlib.pyplot as plt  # 시각화 도구

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/')

transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
train_dataset = torchvision.datasets.ImageFolder(root=r'D:\AI 공부\2. mnist\trainingSample\trainingSample', transform=transform)

train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
vaildloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

first, second = train_dataset[435]  # 특정 index 에 대한 수치 값 ( first : tensor 값, second : key 값 )
print("first type :", type(first), "       first len :", len(first), "         first size :", first.size())
print("second type :", type(second), "      second :", second)

print(len(test_dataset), len(train_dataset))

img = Image.open(r"D:\AI 공부\2. mnist\trainingSet\trainingSet\2\img_14991.jpg")
tras_tensor = transforms.ToTensor()
img_tensor = tras_tensor(img)
print(img_tensor.size())
print(img)


class Model(nn.Module):  # 모델
    def __init__(self):
        super().__init__()  # 모델 연산 정의
        self.Linear = nn.Linear(784, 10, bias=True)

    def forward(self, x):  # 모델 연산의 순서를 정의
        return self.Linear(x)


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.004)

loss_ = []  # 그래프를 그리기 위한 loss 저장용 리스트

n = len(trainloader)
print(n)

for epoch in range(1):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, values = data  # data에는 X, Y가 들어있다.

        inputs = inputs.view(-1, 28 * 28)

        optimizer.zero_grad()  # 최적화 초기화

        outputs = model(inputs)  # 모델에 입력값 대입 후 예측값 산출
        print(outputs, values)
        loss = criterion(outputs, values)  # 손실 함수 계산

        writer.add_scalar("Loss/train", loss, epoch)

        loss.backward()  # 손실 함수 기준으로 역전파 설정
        optimizer.step()  # 역전파를 진행하고 가중치 업데이트 / 최적화
        running_loss += loss.item()  # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.

    if epoch % 20 == 0:
        print(f'epoch : {epoch}, Loss : {running_loss / n}')
    loss_.append(running_loss / n)  # MSE(Mean Squared Error) 계산
writer.flush()
writer.close()

print('Finished Training')
plt.plot(loss_)
plt.title("Training Loss")
plt.xlabel("epoch")


# plt.show()
def evaluation(dataloader):
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()  # 평가를 할 때에는 .eval() 반드시 사용해야 한다.
        for data in dataloader:
            inputs, values = data
            inputs = inputs.view(-1, 28 * 28)
            outputs = model(inputs)
            _, outputs = torch.max(outputs, 1)
            print(outputs)
            print(values)
            total += values.size(0)
            correct += (outputs == values).sum()

    print(100 * correct / total)


vaild = evaluation(vaildloader)

with torch.no_grad():
    model.eval()
    img = img_tensor.view(-1, 28 * 28)
    outputs = model(img)
    outputs = torch.argmax(outputs, 1).item()

    img = img_tensor.view(28, 28)

    plt.title(f'label : {outputs}')
    plt.imshow(img, 'gray')
    plt.show()
