import time

import torchvision.datasets
import torchvision.transforms as transforms

# ANN
import torch
import tqdm as tqdm
from torch import nn, optim  # torch 내의 세부적인 기능을 불러온다. (신경망 기술, 손실함수, 최적화 방법 등)
from torch.utils.data import DataLoader  # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리
import torch.nn.functional as F  # torch 내의 세부적인 기능을 불러온다. (신경망 기술 등)

from torch.utils.tensorboard import SummaryWriter


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

learning_rate = 0.0005
batch_size = 10
epoch_size = 10
weight_decay = 1e-7
project_name = 'mnist'

data_path_train = r'D:\AI_study\ann\2. mnist\trainingSample\trainingSample'
# data_path_test = r'D:\AI 공부\2. mnist\trainingSample\trainingSample'
model_save_path = r'D:\AI_study\ann\2. mnist\model'

pretrained_model = 0  # 0 일 때는 사전 학습 없음, 1일때 사전 학습 있음
model_name = 'temp.pt'

tensorboard_file_name = f"{time.strftime('%H%M%S')}_epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}"
writer = SummaryWriter(log_dir=f'{project_name}/{tensorboard_file_name}', filename_suffix=tensorboard_file_name,
                       comment=f"epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}")


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(),
                                transforms.Normalize((0.0467,), (0.1349,))])    # 데이터 정규화


train_dataset = torchvision.datasets.ImageFolder(root=data_path_train, transform=transform)

# test_dataset = torchvision.datasets.ImageFolder(root=data_path_test, transform=transform)

train_size = int(0.8 * len(train_dataset))
vaild_size = int((len(train_dataset) - train_size) * 0.5)
test_size = int((len(train_dataset) - train_size) * 0.5)

train_dataset, vaild_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, vaild_size, test_size])
print(len(train_dataset))

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
vaildloader = DataLoader(vaild_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print(len(trainloader), len(vaildloader), len(testloader))


class Model(nn.Module):  # 모델
    def __init__(self):
        super().__init__()  # 모델 연산 정의
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):  # 모델 연산의 순서를 정의
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

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
    ls = 2

loss_ = []  # loss 값 저장용

for epoch in range(epoch_size):

    train_loss = 0.0
    vaild_loss = 0.0
    train_acc = 0.0
    vaild_acc = 0.0

    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, values = data  # data에는 X, Y가 들어있다.

        optimizer.zero_grad()  # 최적화 초기화

        outputs = model(inputs)  # 모델에 입력값 대입 후 예측값 산출
        loss = criterion(outputs, values)  # 손실 함수 계산

        loss.backward()  # 손실 함수 기준으로 역전파 설정
        optimizer.step()  # 역전파를 진행하고 가중치 업데이트 / 최적화

        _, outputs = torch.max(outputs, 1)
        train_acc += (outputs == values).sum()
        train_loss += loss.item()  # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.

    model.eval()  # 평가를 할 때에는 .eval() 반드시 사용해야 한다.
    with torch.no_grad():
        for j, data in enumerate(vaildloader, 0):
            inputs, values = data
            outputs = model(inputs)
            loss = criterion(outputs, values)
            vaild_loss += loss.item()
            _, outputs = torch.max(outputs, 1)

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

    # epoch 당 loss, 성능 출력
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

checkpoint = torch.load(fr'{model_save_path}\{model_name}')
model.load_state_dict(checkpoint['model'])
print(model)
model.eval()

with torch.no_grad():
    for i, data in enumerate(test_dataset, 0):
        inputs, values = data
        test_output = model(inputs)
        _, outputs = torch.max(test_output, 1)
        acc += (outputs == values).sum()
        print(f'예측 {outputs} \n실제 값 {values}')
    print(100 * acc / len(test_dataset))
