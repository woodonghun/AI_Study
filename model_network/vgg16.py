from torch import nn  # torch 내의 세부적인 기능을 불러온다. (신경망 기술, 손실함수, 최적화 방법 등)
import torch.nn as F  # torch 내의 세부적인 기능을 불러온다. (신경망 기술 등)


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
        self.fc3 = nn.Linear(4096, 8)

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
