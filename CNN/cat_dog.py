import time

import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
# ANN
import torch
from torchvision import models
from tqdm import tqdm
from torch import nn, optim  # torch 내의 세부적인 기능을 불러온다. (신경망 기술, 손실함수, 최적화 방법 등)
from torch.utils.data import DataLoader  # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리
import torch.nn as F  # torch 내의 세부적인 기능을 불러온다. (신경망 기술 등)

from torch.utils.tensorboard import SummaryWriter
import feature_map_show

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

learning_rate = 0.00001
batch_size = 4
epoch_size = 2
weight_decay = 5e-7
project_name = 'cat_dog'

data_path_train = r'D:\AI_study\sample\train'
# data_path_train = r'D:\AI_study\cnn\2catdog\cat_dog\training_set'
model_save_path = r'D:\AI_study\cnn\2catdog\model'

pretrained_model = 0  # 0 일 때는 사전 학습 없음, 1일때 사전 학습 있음
model_name = 'quantresnet__.pt'

tensorboard_file_name = f"{time.strftime('%H%M%S')}_epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}"
writer = SummaryWriter(log_dir=f'{project_name}/{tensorboard_file_name}', filename_suffix=tensorboard_file_name,
                       comment=f"epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}")

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48827466, 0.4551035, 0.41741803), (0.05540232, 0.054113153, 0.055464733))
])  # 데이터 정규화
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4865031, 0.4527351, 0.41355005), (0.05324953, 0.052485514, 0.054207902))
])  # 데이터 정규화

train_dataset = torchvision.datasets.ImageFolder(root=data_path_train, transform=transform_train)
test_dataset = torchvision.datasets.ImageFolder(root=r'D:\AI_study\cnn\2catdog\cat_dog\test_set', transform=transform_test)

train_size = int(0.8 * len(train_dataset))
vaild_size = len(train_dataset) - train_size
# test_size = int((len(train_dataset) - train_size) // 2)
print(len(train_dataset), train_size, vaild_size)
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, vaild_size])
print(len(train_dataset), len(valid_dataset), len(test_dataset))

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

print(len(trainloader), len(validloader), len(testloader))


class VGGNet16(nn.Module):  # 모델
    def __init__(self):
        super().__init__()  # 모델 연산 정의
        # input = 224x224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)  # 224x224x64
        self.relu1 = F.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112x112x64

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)  # 112x112x128
        self.relu2 = F.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56x56x128

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)  # 56x56x256
        self.relu3 = F.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu4 = F.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28x256

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)  # 28x28x512
        self.relu5 = F.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu6 = F.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14x512

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)  # 14x14x512
        self.relu7 = F.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu8 = F.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7x512

        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.relu9 = F.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu10 = F.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, 2)

    def forward(self, x):  # 모델 연산의 순서를 정의
        x = self.conv1(x)  # conv1 -> ReLU -> pool1
        x = self.relu1(x)
        x = self.pool1(x)  # conv1 -> ReLU -> pool1

        x = self.conv2(x)  # conv1 -> ReLU -> pool1
        x = self.relu2(x)
        x = self.pool2(x)  # conv1 -> ReLU -> pool1

        x = self.conv3(x)  # conv1 -> ReLU -> pool1
        x = self.relu3(x)
        x = self.conv4(x)  # conv1 -> ReLU -> pool1
        x = self.relu4(x)
        x = self.pool3(x)  # conv1 -> ReLU -> pool1

        x = self.conv5(x)  # conv1 -> ReLU -> pool1
        x = self.relu5(x)
        x = self.conv6(x)  # conv1 -> ReLU -> pool1
        x = self.relu6(x)
        x = self.pool4(x)  # conv1 -> ReLU -> pool1

        x = self.conv7(x)  # conv1 -> ReLU -> pool1
        x = self.relu7(x)
        x = self.conv8(x)  # conv1 -> ReLU -> pool1
        x = self.relu8(x)
        x = self.pool5(x)  # conv1 -> ReLU -> pool1

        x = x.view(-1, 7 * 7 * 512)
        x = self.fc1(x)
        x = self.relu9(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu10(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class Trainer:
    def __init__(self, model, trainloader, validloader, learning_rate, weight_decay,
                 epoch_size, model_save_path, model_name, pretrained_model=None):
        self.trainloader = trainloader
        self.validloader = validloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epoch_size = epoch_size
        self.pretrained_model = pretrained_model
        self.model_save_path = model_save_path
        self.model_name = model_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # gpu 설정


        self.model = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.pretrained_model:
            checkpoint = torch.load(fr'{self.model_save_path}\{self.model_name}')
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.loss_ = checkpoint['loss']
            self.ep = checkpoint['epoch']
            self.ls = self.loss_[-1]  # 제일 마지막 로스값
            print(f"epoch={self.ep}, loss={self.ls}")
            self.ep += 1
        else:
            self.ep = 0
            self.ls = 2

        self.loss_ = []  # loss 값 저장용
        self.writer = SummaryWriter()

    def train(self):

        for epoch in range(self.epoch_size):
            train_loss = 0.0
            valid_loss = 0.0
            train_acc = 0.0
            valid_acc = 0.0

            self.model.train()

            # progressbar 설정
            trainloader_tqdm = tqdm(enumerate(self.trainloader, 0), total=len(self.trainloader),
                                    desc=f'train-epoch : (X/X), loss : X, acc : X', ncols=100, leave=True)

            for i, data in trainloader_tqdm:
                inputs, values = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()  # 최적화 초기화
                outputs = self.model(inputs)  # 모델에 입력값 대입 후 예측값 산출
                loss = self.criterion(outputs, values)  # 손실 함수 계산
                loss.backward()  # 손실 함수 기준으로 역전파 설정
                self.optimizer.step()  # 역전파를 진행하고 가중치 업데이트 / 최적화

                _, outputs = torch.max(outputs, 1)  # outputs 중에서 가장 큰 값의 index 가 결과
                train_acc += (outputs == values).sum()  # 결과가 같으면 train_acc 에 합함
                train_loss += loss.item()  # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.

                # progressbar 앞부분 변경
                trainloader_tqdm.set_description(f'train-epoch : ({epoch + 1}/{self.epoch_size}),'
                                                 f' loss : {train_loss / (i + 1):.4f},'
                                                 f' acc : {100 * train_acc / ((i + 1) * self.trainloader.batch_size):.4f}')

            self.model.eval()  # 평가를 할 때에는 .eval() 반드시 사용해야 한다.
            with torch.no_grad():
                validloader_tqdm = tqdm(enumerate(self.validloader, 0), total=len(self.validloader),
                                        desc=f'valid-epoch : (X/X), loss : X, acc : X', ncols=100, leave=True)

                for j, data in validloader_tqdm:
                    inputs, values = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, values)
                    valid_loss += loss.item()
                    _, outputs = torch.max(outputs, 1)

                    valid_acc += (outputs == values).sum()
                    validloader_tqdm.set_description(f'valid-epoch : ({epoch + 1}/{self.epoch_size}),'
                                                     f' loss : {valid_loss / (j + 1):.4f},'
                                                     f' acc : {100 * valid_acc / ((j + 1) * self.validloader.batch_size):.4f}')

            loss_save = valid_loss / len(self.validloader)  # 모델 저장
            self.loss_.append(loss_save)
            # loss 값이 작아질 때 마다 저장
            # if loss_save < self.ls:
            torch.save({'epoch': epoch,
                            'loss': self.loss_,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()
                            }, fr'{self.model_save_path}\{self.model_name}')

            # Tensorboard 에 저장
            writer.add_scalar("train/acc", 100 * train_acc / (len(self.trainloader) * self.trainloader.batch_size), epoch)
            writer.add_scalar("train/loss", train_loss / len(self.trainloader), epoch)
            writer.add_scalar('valid/acc', 100 * valid_acc / (len(self.validloader) * self.validloader.batch_size), epoch)
            writer.add_scalar('valid/loss', valid_loss / len(self.validloader), epoch)
            writer.add_scalars('loss', {'train': train_loss / len(self.trainloader), 'valid': valid_loss / len(self.validloader)}, epoch)
            writer.add_scalars('acc', {'train': 100 * train_acc / (len(self.trainloader) * self.trainloader.batch_size),
                                       'valid': 100 * valid_acc / (len(self.validloader) * self.validloader.batch_size)}, epoch)

            # epoch 당 loss, 성능 출력
            # if epoch % 10 == 9:
            print(
                f" epoch {epoch + 1} -"
                f" train loss: {train_loss / len(self.trainloader):.4f},"
                f" train acc : {100 * train_acc / (len(self.trainloader) * self.trainloader.batch_size):.4f}"
                f" ----------"
                f" valid loss : {valid_loss / len(self.validloader):.4f}"
                f" valid acc : {100 * valid_acc / (len(self.validloader) * self.validloader.batch_size):.4f}"
            )

        writer.flush()
        writer.close()


class Predict:
    def __init__(self, model, model_save_path, model_name, testloader, test_dataset):
        self.acc = 0.0
        self.device = torch.device('cpu')
        self.model = model
        self.checkpoint = torch.load(fr'{model_save_path}\{model_name}', map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.eval()
        self.testloader = testloader
        self.test_dataset = test_dataset

    def evaluate(self):
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.testloader, 0), total=len(self.testloader), ncols=100, leave=True):
                inputs, values = data[0], data[1]
                test_output = self.model(inputs)
                _, outputs = torch.max(test_output, 1)
                self.acc += (outputs == values).sum()
            print(100 * self.acc / len(self.test_dataset))


if __name__ == "__main__":
    model = models.resnet50(pretrained=True).to(device)
    # model = models.vgg11(pretrained=True).to('cpu')

    # model = VGGNet16().to(device)

    # train
    train_vgg = Trainer(model, trainloader, validloader, learning_rate, weight_decay, epoch_size, model_save_path, model_name)
    train_vgg.train()

    # predict
    # predict_model = Predict(model, model_save_path, model_name, testloader, test_dataset)
    # predict_model.evaluate()

    # feature map
    # fms = feature_map_show.FeatureMapVisualizer(model)
    # fms.visualize(test_dataset[1][0].unsqueeze(0),
    #               {'conv1': [0, 32, 2, 51], 'conv2': [1, 2, 64, 4, 5], 'conv4': [1, 15, 3, 48, 110], 'conv8': [5, 164, 484, 115, 31, 21, 12, 44, 84, 99, 0,66, 511]})
