import os
import time
from PIL import Image
from glob import glob
# import numpy as np

import torchvision.transforms as transforms

# ANN
import torch
# from torchvision import models
from tqdm import tqdm
from torch import nn, optim  # torch 내의 세부적인 기능을 불러온다. (신경망 기술, 손실함수, 최적화 방법 등)
from torch.utils.data import DataLoader  # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리
# import torch.nn as F  # torch 내의 세부적인 기능을 불러온다. (신경망 기술 등)

from torch.utils.tensorboard import SummaryWriter
from feature_map_show import FeatureMapVisualizer

import model_network.resnet as res
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

learning_rate = 0.0001
batch_size = 32
epoch_size = 10
weight_decay = 5e-7
project_name = '8class'

data_path_train = r'D:\AI_study\cnn\3. 8종 이미지 분류\train_natural_images'
data_path_test = r'D:\AI_study\cnn\3. 8종 이미지 분류\test_natural_images'
model_save_path = r'D:\AI_study\cnn\3. 8종 이미지 분류\model'

feature_map = False
feature_map_layer_name = None  # feature map 을 저장할 layer, map index dict {'conv1': [1, 2, 3, 4, 5], 'layer1.2.con2:[1,2,3,4,5]}
feature_map_save_epoch = 2  # feature map 을 저장할 epoch의 배수   ex) 2 이면 2, 4, 6, 8... 일때 폴더 생성
feature_map_save_path = r'C:\woo_project\AI_Study\sample_data'  # 피쳐맵 이미지 폴더를 생성할 경로, 피쳐맵 폴더 이름은 feature_map 으로 고정


pretrained_model = 0  # 0 일 때는 사전 학습 없음, 1일때 사전 학습 있음
model_name = 'resnet18_8class_.pt'

tensorboard_file_name = f"{time.strftime('%H%M%S')}_epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}"


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir

        self.classes = os.listdir(self.root_dir)
        self.transforms = transforms
        self.data = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.root_dir, cls)
            for img in glob(os.path.join(cls_dir, '*.jpg')):
                self.data.append(img)
                self.labels.append(idx)

    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.labels[idx]
        img = Image.open(img_path)
        file_name = img_path.split("\\")[-1]

        if self.transforms:
            img = self.transforms(img)

        sample = {'image': img, 'label': label, 'filename': file_name}

        return sample

    def __len__(self):
        return len(self.data)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5088954, 0.46383116, 0.41928688), (0.05792271, 0.062752955, 0.07355038))
])  # 데이터 정규화
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.52701133, 0.47480434, 0.42878655), (0.06052063, 0.063967526, 0.075863875))
])  # 데이터 정규화

train_dataset = CustomDataset(data_path_train, transform_train)
test_dataset = CustomDataset(data_path_test, transform_test)

train_size = int(0.8 * len(train_dataset))
vaild_size = len(train_dataset) - train_size
# test_size = int((len(train_dataset) - train_size) // 2)
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, vaild_size])
print(len(train_dataset), len(valid_dataset), len(test_dataset))

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

print(len(trainloader), len(validloader), len(testloader))

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
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 8], gamma=0.5)

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

    def train(self):
        writer = SummaryWriter(log_dir=f'{project_name}/{tensorboard_file_name}', filename_suffix=tensorboard_file_name,
                               comment=f"epoch={epoch_size}_lr={learning_rate}_batch_size={batch_size}")
        for epoch in range(self.epoch_size):
            visualizer = FeatureMapVisualizer(self.model, feature_map_save_path, feature_map_save_epoch, use=feature_map)   # feature map 생성 선언
            visualizer.create_feature_map_epoch_folder(epoch)  # feature map 폴더 안 epoch 폴더 생성

            train_loss = 0.0
            valid_loss = 0.0
            train_acc = 0.0
            valid_acc = 0.0

            self.model.train()

            # progressbar 설정
            trainloader_tqdm = tqdm(enumerate(self.trainloader, 0), total=len(self.trainloader),
                                    desc=f'train-epoch : (X/X), loss : X, acc : X', ncols=100, leave=True)

            for i, data in trainloader_tqdm:
                inputs, values = data['image'].to(self.device), data['label'].to(self.device)
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
                if i == 1:
                    visualizer.visualize(epoch, inputs, data['filename'], feature_map_layer_name)  # feature_map - epoch 폴더 안에 생성

            self.model.eval()  # 평가를 할 때에는 .eval() 반드시 사용해야 한다.
            with torch.no_grad():
                validloader_tqdm = tqdm(enumerate(self.validloader, 0), total=len(self.validloader),
                                        desc=f'valid-epoch : (X/X), loss : X, acc : X', ncols=100, leave=True)

                for j, data in validloader_tqdm:
                    inputs, values = data['image'].to(self.device), data['label'].to(self.device)
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

            self.scheduler.step()

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
                inputs, values = data['image'], data['label']
                test_output = self.model(inputs)
                _, outputs = torch.max(test_output, 1)
                self.acc += (outputs == values).sum()
            print(100 * self.acc / len(self.test_dataset))


if __name__ == "__main__":
    # model = models.resnet18(pretrained=True).to(device)
    # model = models.vgg11(pretrained=True).to('cpu')

    # model = res.resnet18().to(device)
    model = res.resnet18().to('cpu')
    print(model)
    # train
    # train_vgg = Trainer(model, trainloader, validloader, learning_rate, weight_decay, epoch_size, model_save_path, model_name)
    # train_vgg.train()

    # predict
    predict_model = Predict(model, model_save_path, model_name, testloader, test_dataset)
    predict_model.evaluate()

    # feature map
    # fms = feature_map_show.FeatureMapVisualizer(model)
    # fms.visualize(1, test_dataset[0][0].unsqueeze(0),
    #               {'conv1': [0, 32, 2, 51, 3, 12], 'conv2': [1, 2, 64, 4, 5], 'conv4': [1, 15, 3, 48, 110],
    #                'conv8': [5, 164, 484, 115, 31, 21, 12, 44, 84, 99, 0, 66, 511]})
