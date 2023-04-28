import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader  # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리

import matplotlib.pyplot as plt
import os
import cv2
from glob import glob
import numpy as np
from torchvision.models import resnet18


class GradCam(nn.Module):
    cam = []
    layer_target_index = []

    def __init__(self, model, module):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.module = module
        self.register_hooks()
        GradCam.layer_target_index.append(str(module))

    def register_hooks(self):
        save_name = []
        i = 0
        for name, layer in self.model.named_modules():  # module 정보
            i += 1
            if isinstance(layer, nn.Conv2d):  # layer 와 conv2 에 해당되는 것만 가지고옴
                save_name.append(name)  # layer 이름 저장
        for name, layer in self.model.named_modules():
            # print(name,layer)

            if isinstance(layer, nn.Conv2d) and name == self.module:  # 저장한 layer 중 첫번 째 마지막 layer 만 hook

                layer.register_forward_hook(self.forward_hook)
                layer.register_backward_hook(self.backward_hook)

    def forward(self, input, target_index):
        outs = self.model(input.to(self.device))
        outs = outs.squeeze()  # [1, num_classes]  --> [num_classes]

        # 가장 큰 값을 가지는 것을 target index 로 사용
        if target_index is None:
            target_index = outs.argmax()

        outs[target_index].backward(retain_graph=True)
        a_k = torch.mean(self.backward_result, dim=(1, 2), keepdim=True)  # [512, 1, 1]
        out = torch.sum(a_k * self.forward_result, dim=0).cpu()  # [512, 7, 7] * [512, 1, 1]
        out = torch.relu(out) / torch.max(out)  # 음수를 없애고, 0 ~ 1 로 scaling # [7, 7]
        out = F.upsample_bilinear(out.unsqueeze(0).unsqueeze(0), [224, 224])  # 4D로 바꿈
        return out.cpu().detach().squeeze().numpy()

    def forward_hook(self, _, input, output):
        self.forward_result = torch.squeeze(output)

    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])


def show_cam_on_image(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[:, :, ::-1]  # matplot 과 cv2 는 rgb 채널이 달라서 변경해줘야 한다. cv2 bgr, plt rgb
    # print(img)
    # cam = heatmap + np.float32(img)
    # cam = cam / np.max(cam)
    GradCam.cam.append(heatmap)


def make_plt_cam(cam, image, epoch):
    fig = plt.figure(figsize=(10, 5))
    row = len(cam) // 10 + 1
    if len(cam) > 9:  # 10 이하의 이미지 개수에 맞춰서 column 의 길이 조정 하기 위해 사용
        column_size = 10
    else:
        column_size = len(cam)

    for i, j in enumerate(cam):
        ax = fig.add_subplot(row, column_size, i + 1)  # row,column,index =>ex) 3,3,2 => 3x3 의 2번째 index
        ax.imshow(image, cmap='gray')  # ?? cmap='gray' 지우면 색상 나옴
        ax.imshow(j, cmap='gray', alpha=0.6)  # ?? cmap='gray' 지우면 색상 나옴
        ax.axis('off')  # grid 제거
        ax.set_title(f"{GradCam.layer_target_index[i]}", fontsize=7)  # subplot name
    fig.suptitle(f'epoch - {epoch}')

    # if len(input_name) != 1:  # batch size 가 1 이상일때 아닐때 title 명 조정
    #     fig.suptitle(f'{name}\n{input_name[0]}')
    # else:
    #     fig.suptitle(f'{name}\n{input_name[-1]}')
    plt.savefig(fr'C:\Users\3DONS\Desktop\새 폴더/grad_cam_{epoch}.png')

    # plt.show()  # plt.save 랑 같이 사용 불가능 / show 사용시 모든 os.mkdir 제거하고 하면 편함 ( 안해도 됨 )


def insert_input_module_layer(model, train_dataset, epoch, module_layer: list):
    GradCam.cam = []
    # data, label, label_list = train_dataset[0]['image'], train_dataset[0]['label_name'], train_dataset[0]['label_list']  # 이미지 한 장과 라벨 불러오기
    # print(train_dataset[0]['filename'])

    data, label = train_dataset[0], train_dataset[1]  # 이미지 한 장과 라벨 불러오기

    data.unsqueeze_(0)  # 4차원 3차원 [피쳐수 ,너비, 높이] -> [1,피쳐수 ,너비, 높이]  ???
    original_img = (data[0][0].cpu().numpy())

    for i, k in enumerate(module_layer):
        grad_cam = GradCam(model=model, module=k)
        mask = grad_cam(data, None)
        show_cam_on_image(mask)
    make_plt_cam(GradCam.cam, original_img, epoch)


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

        sample = {'image': img, 'label': label, 'filename': file_name, 'label_name': self.classes[label], 'label_list': self.classes}

        return sample

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path_train = r'D:\AI_study\cnn\2catdog\cat_dog\111'
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])  # 데이터 정규화

    train_dataset = CustomDataset(data_path_train, transform_train)
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)

    model = resnet18(pretrained=True)

    model.eval()

    data, label, label_list = train_dataset[0]['image'], train_dataset[0]['label_name'], train_dataset[0]['label_list']  # 이미지 한 장과 라벨 불러오기

    data.unsqueeze_(0)  # 4차원 3차원 [피쳐수 ,너비, 높이] -> [1,피쳐수 ,너비, 높이]  ???
    original_img = (data[0][0].cpu().numpy())

    # original_img = np.transpose(original_img, (1, 2, 0))

    insert_input_module_layer(model, train_dataset, 0, ['layer1.0.conv1', 'layer1.0.conv2', 'layer2.0.conv1', 'layer1.0.conv2'])
