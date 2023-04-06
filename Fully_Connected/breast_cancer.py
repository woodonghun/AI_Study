import pandas as pd  # 데이터프레임 형태를 다룰 수 있는 라이브러리

# ANN
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim  # torch 내의 세부적인 기능을 불러온다. (신경망 기술, 손실함수, 최적화 방법 등)
from torch.utils.data import DataLoader, Dataset, random_split  # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리

from torch.utils.tensorboard import SummaryWriter

import time

'''
    tensorboard 사용시 Terminal 에서 tensorboard --logdir=./~~~
    logdir => log_dir 에서 작성한 경로의 상위 폴더
'''
learning_rate = 0.01
batch_size = 5
epoch_size = 700
weight_decay = 1e-7
project_name = 'breast_cancer'

model_save_path = r'D:\AI 공부\5. 유방암 진단\model'

pretrained_model = 0  # 0 일 때는 사전 학습 없음, 1일때 사전 학습 있음
model_name = 'temp.pt'

''''''''''''''''''''''''''''''''''''''

tensorboard_file_name = f"{time.strftime('%H%M%S')}_epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}"

writer = SummaryWriter(log_dir=f'{project_name}/{tensorboard_file_name}', filename_suffix=tensorboard_file_name,
                       comment=f"epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}")


class CustomDataset(Dataset):
    def __init__(self, file_path):
        scaler_minmax = MinMaxScaler()

        dataframe = pd.read_csv(file_path)
        self.y = dataframe['diagnosis']  # target
        self.y = self.y.replace(to_replace=['M', 'B'], value=[1, 0])  # M = 악성, B = 양성, 악성 양성 예측
        self.y = self.y.to_numpy().reshape((-1, 1))
        self.y = torch.FloatTensor(self.y)

        self.x = dataframe.drop(['id', 'diagnosis'], axis=1)  # data
        scaler_minmax.fit(self.x)
        self.x = scaler_minmax.transform(self.x)
        self.x = torch.FloatTensor(self.x)

        self.len = len(dataframe)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


dataset = CustomDataset(r'D:\AI 공부\5. 유방암 진단\data.csv')  # 커스텀데이터셋 생성
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


class FullyConnected(nn.Module):  # 모델
    def __init__(self):
        super().__init__()  # 모델 연산 정의
        self.fc1 = nn.Linear(30, 1, bias=True)  # 편향

    def forward(self, x):  # 모델 연산의 순서를 정의
        x = torch.sigmoid(self.fc1(x))
        # BCE loss 는 0 ~ 1 사이의 값으로 출력됨
        # 따라서 BCE 사용시 마지막 시그모이드,
        return x


# lr은 학습률이다.
# weight_decay는 L2 정규화에서의 penalty 정도를 의미한다.
model = FullyConnected()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 학습모델이 존재할 경우
if pretrained_model == 1:
    checkpoint = torch.load(fr'{model_save_path}\{model_name}')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_ = checkpoint['loss']
    ep = checkpoint['epoch']
    ls = loss_[-1]
    print(f"epoch={ep}, loss={ls}")
    ep = ep + 1
else:
    ep = 0
    ls = 1

loss_ = []  # loss 값 저장용
for epoch in range(epoch_size):

    running_train_loss = 0.0
    test_accuracy = 0.0
    vaild_accuracy = 0.0
    running_vaild_loss = 0.0

    # train
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, values = data  # data에는 X, Y가 들어있다.

        optimizer.zero_grad()  # 최적화 초기화

        test_outputs = model(inputs)  # 모델에 입력값 대입 후 예측값 산출

        loss = criterion(test_outputs, values)  # 손실 함수 계산

        loss.backward()  # 손실 함수 기준으로 역전파 설정
        optimizer.step()  # 역전파를 진행하고 가중치 업데이트 / 최적화

        test_output = torch.round(test_outputs)  # 반올림
        test_accuracy += (values == test_output).sum()

        running_train_loss += loss.item()  # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.

    # 모델 저장
    loss_save = running_train_loss / len(trainloader)
    loss_.append(loss_save)

    # loss 값이 작아질 때 마다 저장
    if loss_save < ls:
        torch.save({'epoch': epoch,
                    'loss': loss_,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, fr'{model_save_path}\{model_name}')

    # vaild
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(vaildloader, 0):
            inputs, values = data
            vaild_output = model(inputs)
            loss = criterion(vaild_output, values)
            running_vaild_loss += loss.item()
            vaild_output = torch.round(vaild_output)  # 반올림
            vaild_accuracy += (values == vaild_output).sum()

    # Tensorboard 에 저장
    writer.add_scalar("train/acc", 100 * test_accuracy / len(train_dataset), epoch)
    writer.add_scalar("train/loss", running_train_loss / len(trainloader), epoch)
    writer.add_scalar('vaild/acc', 100 * vaild_accuracy / len(vaild_dataset), epoch)
    writer.add_scalar('vaild/loss', running_vaild_loss / len(vaildloader), epoch)
    writer.add_scalars('loss', {'train': running_train_loss / len(trainloader), 'vaild': running_vaild_loss / len(vaildloader)}, epoch)
    writer.add_scalars('acc', {'train': 100 * test_accuracy / len(train_dataset), 'vaild': 100 * vaild_accuracy / len(vaild_dataset)}, epoch)

    if epoch % 10 == 9:
        print(
        f"epoch {epoch + 1} -"
        f" test loss: {running_train_loss / len(trainloader):.4f},"
        f" test acc : {100 * test_accuracy / len(train_dataset):.4f}"
        f" ////////////////////"
        f" vaild loss : {running_vaild_loss / len(vaildloader):.4f}"
        f" vaild acc : {100 * vaild_accuracy / len(vaild_dataset):.4f}"
    )

writer.flush()
writer.close()

checkpoint = torch.load(fr'{model_save_path}\{model_name}')
model.load_state_dict(checkpoint['model'])
# testData 로 예측
model.eval()

acc = 0.0
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, values = data
        test_output = model(inputs)
        test_output = torch.round(test_output)
        print(f'예측 {test_output.tolist()} \n실제 값 {values.tolist()}')
        acc += (values == test_output).sum()
    print(100 * acc / len(test_dataset))
