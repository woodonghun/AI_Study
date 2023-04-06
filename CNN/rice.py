import time

import torchvision.datasets
import torchvision.transforms as transforms
from torchvision import models
# ANN
import torch
from tqdm import tqdm
from torch import nn, optim  # torch 내의 세부적인 기능을 불러온다. (신경망 기술, 손실함수, 최적화 방법 등)
from torch.utils.data import DataLoader  # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리
import torch.nn.functional as F  # torch 내의 세부적인 기능을 불러온다. (신경망 기술 등)

from torch.utils.tensorboard import SummaryWriter

from time import sleep

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

learning_rate = 0.0001
batch_size = 32
epoch_size = 10
weight_decay = 1e-7
project_name = 'rice'

data_path_train = r'D:\AI_study\cnn\1. 쌀 분류\rice_train'
model_save_path = r'D:\AI_study\cnn\1. 쌀 분류\model'

pretrained_model = 0  # 0 일 때는 사전 학습 없음, 1일때 사전 학습 있음
model_name = 'temp.pt'

tensorboard_file_name = f"{time.strftime('%H%M%S')}_epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}"
writer = SummaryWriter(log_dir=f'{project_name}/{tensorboard_file_name}', filename_suffix=tensorboard_file_name,
                       comment=f"epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}")

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

transform = transforms.Compose([
    transforms.Resize((100, 100)),  # 동일하게 image 사이즈 변경
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    # 데이터 정규화

train_dataset = torchvision.datasets.ImageFolder(root=data_path_train, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=r'D:\AI_study\cnn\1. 쌀 분류\rice_test', transform=transform)


train_size = int(0.8 * len(train_dataset))
# valid_size = int((len(train_dataset) - train_size)*0.5)
valid_size = len(train_dataset) - train_size

# test_size = int((len(train_dataset) - train_size)*0.5)

# train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size, test_size])
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

print(len(train_dataset), len(valid_dataset), len(test_dataset))

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

print(len(trainloader), len(validloader), len(testloader))


class Model(nn.Module):  # 모델
    def __init__(self):
        super().__init__()  # 모델 연산 정의
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, padding=1, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, padding=1, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(num_features=20)
        # self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, padding=1, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(32 * 25 * 25, 5)

    def forward(self, x):  # 모델 연산의 순서를 정의
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 32 * 25 * 25)
        x = self.fc1(x)

        return x


model = Model().to(device)  # .to(device) => GPU 사용
# model = Model()
# model = models.resnet50(pretrained=True).to(device)
# model = models.resnet50(pretrained=True)
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

# epoch_tqdm = tqdm(range(epoch_size), total=epoch_size, desc=f'train-loss : X, valid-loss : X, train_acc : X, valid_acc', ncols=100, leave=True, position=0)
for epoch in range(epoch_size):

    train_loss = 0.0
    valid_loss = 0.0
    train_acc = 0.0
    valid_acc = 0.0

    model.train()

    # progressbar 설정
    trainloader_tqdm = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f'train-epoch : (X/X), loss : X, acc : X', ncols=100, leave=True)

    # for i, data in enumerate(trainloader, 0):
    for i, data in trainloader_tqdm:
        inputs, values = data[0].to(device), data[1].to(device)  # data에는 X, Y가 들어있다.
        # inputs, values = data[0], data[1]  # data에는 X, Y가 들어있다.
        optimizer.zero_grad()  # 최적화 초기화

        outputs = model(inputs)  # 모델에 입력값 대입 후 예측값 산출

        loss = criterion(outputs, values)  # 손실 함수 계산

        loss.backward()  # 손실 함수 기준으로 역전파 설정
        optimizer.step()  # 역전파를 진행하고 가중치 업데이트 / 최적화

        _, outputs = torch.max(outputs, 1)
        train_acc += (outputs == values).sum()
        train_loss += loss.item()  # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.

        # progressbar 앞부분 변경
        trainloader_tqdm.set_description(f'train-epoch : ({epoch + 1}/{epoch_size}),'
                                         f' loss : {train_loss / (i + 1):.4f},'
                                         f' acc : {100 * train_acc / ((i + 1)*batch_size):.4f}')
        # epoch_tqdm.set_description(f'train-loss : {train_loss / len(trainloader):.4f}, valid-loss : X, train_acc : {100 * train_acc / len(train_dataset):.4f},
        # valid_acc')

    model.eval()  # 평가를 할 때에는 .eval() 반드시 사용해야 한다.
    with torch.no_grad():
        validloader_tqdm = tqdm(enumerate(validloader, 0), total=len(validloader), desc=f'valid-epoch : (X/X), loss : X, acc : X', ncols=100, leave=True)
        # for i, data in enumerate(validloader, 0):
        for j, data in validloader_tqdm:
            inputs, values = data[0].to(device), data[1].to(device)
            # inputs, values = data[0], data[1]

            outputs = model(inputs)
            loss = criterion(outputs, values)
            valid_loss += loss.item()
            _, outputs = torch.max(outputs, 1)

            valid_acc += (outputs == values).sum()
            validloader_tqdm.set_description(f'valid-epoch : ({epoch + 1}/{epoch_size}),'
                                             f' loss : {valid_loss / (j + 1):.4f},'
                                             f' acc : {100 * valid_acc / ((j + 1)*batch_size):.4f}')
        # epoch_tqdm.set_description(f'train-loss : {train_loss / len(trainloader):.4f}, valid-loss : {valid_loss / len(validloader):.4f},'
        #                            f' train_acc : {100 * train_acc / len(train_dataset):.4f}, valid_acc : {100 * valid_acc / len(valid_dataset):.4f}')

    # 모델 저장
    loss_save = valid_loss / len(valid_dataset)
    loss_.append(loss_save)
    # loss 값이 작아질 때 마다 저장
    if loss_save < ls:
        torch.save({'epoch': epoch,
                    'loss': loss_,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, fr'{model_save_path}\{model_name}')

    # Tensorboard 에 저장
    writer.add_scalar("train/acc", 100 * train_acc / (len(trainloader)*batch_size), epoch)
    writer.add_scalar("train/loss", train_loss / len(trainloader), epoch)
    writer.add_scalar('valid/acc', 100 * valid_acc / (len(validloader)*batch_size), epoch)
    writer.add_scalar('valid/loss', valid_loss / len(validloader), epoch)
    writer.add_scalars('loss', {'train': train_loss / len(trainloader), 'valid': valid_loss / len(validloader)}, epoch)
    writer.add_scalars('acc', {'train': 100 * train_acc / (len(trainloader)*batch_size), 'valid': 100 * valid_acc / (len(validloader)*batch_size)}, epoch)

    # epoch 당 loss, 성능 출력
    # if epoch % 10 == 9:
    print(
            f" epoch {epoch + 1} -"
            f" train loss: {train_loss / len(trainloader):.4f},"
            f" train acc : {100 * train_acc / (len(trainloader)*batch_size):.4f}"
            f" ----------"
            f" valid loss : {valid_loss / len(validloader):.4f}"
            f" valid acc : {100 * valid_acc / (len(validloader)*batch_size):.4f}"
        )

writer.flush()
writer.close()

acc = 0.0

checkpoint = torch.load(fr'{model_save_path}\{model_name}')
model.load_state_dict(checkpoint['model'])
# print(model)
model.eval()

with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, values = data[0].to(device), data[1].to(device)
        # inputs, values = data[0], data[1]
        test_output = model(inputs)
        _, outputs = torch.max(test_output, 1)
        acc += (outputs == values).sum()
        print(f'\n예측 값 {outputs.tolist()} \n실제 값 {values.tolist()}')
    print(100 * acc / len(test_dataset))
