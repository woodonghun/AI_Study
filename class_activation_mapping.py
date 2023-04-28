import os

import numpy as np
from matplotlib import pyplot as plt
import cv2

import torch

import torch.nn as nn

"""
    cam 출력 함수 cam 강의에서 나오는 코드 수정 
"""


class cam:
    CAM_folder_name = 'class activation map'

    def __init__(self, model, path: str, device, image_size=(224, 224)):
        self.device = device
        self.model = model
        self.image_size = image_size
        self.save_path = path + '/' + cam.CAM_folder_name
        self.activation = {}
        if cam.CAM_folder_name not in os.listdir(path):  # 사용 상태이고 path 에 해당 폴더 없으면 생성
            os.mkdir(self.save_path)  # feature_map 저장할 폴더 생성


    # Visualize feature maps

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def cam(self, trainset, img_sample, img_size):
        self.model.eval()
        with torch.no_grad():  # requires_grad 비활성화
            save_name = []
            i = 0
            for name, layer in self.model.named_modules():  # module 정보
                i += 1
                if isinstance(layer, nn.Conv2d):  # layer 와 conv2 에 해당되는 것만 가지고옴
                    save_name.append(name)  # layer 이름 저장
            for name, layer in self.model.named_modules():
                if isinstance(layer, nn.Conv2d) and name == save_name[-1]:  # 저장한 layer 중 첫번 째 마지막 layer 만 hook
                    layer.register_forward_hook(self.get_activation(name))

            data, label, label_list = trainset[img_sample]['image'], trainset[img_sample]['label_name'], trainset[img_sample]['label_list']  # 이미지 한 장과 라벨 불러오기

            data.unsqueeze_(0)  # 4차원 3차원 [피쳐수 ,너비, 높이] -> [1,피쳐수 ,너비, 높이]  ???
            output = self.model(data.to(self.device))
            _, prediction = torch.max(output, 1)
            act = self.activation[save_name[-1]].squeeze()  # 4차원 [1,피쳐수 ,너비, 높이] -> 3차원 [피쳐수 ,너비, 높이]  ???
            w = self.model.fc.weight  # classifer의 가중치 불러오기
            for idx in range(act.size(0)):  # CAM 연산
                if idx == 0:
                    tmp = act[idx] * w[prediction.item()][idx]
                else:
                    tmp += act[idx] * w[prediction.item()][idx]
            # 모든 이미지 팍셀값을 0~255로 스케일하기
            normalized_cam = tmp.cpu().numpy()
            normalized_cam = (normalized_cam - np.min(normalized_cam)) / (np.max(normalized_cam) - np.min(normalized_cam))
            original_img = (data[0][0].cpu().numpy())

            # 원본 이미지 사이즈로 리사이즈
            cam_img = cv2.resize(np.uint8(normalized_cam * 255), dsize=(img_size, img_size))
            predict_label = label_list[prediction]
        return cam_img, original_img, label, predict_label

    def plot_cam(self, epoch, trainset, img_size, start):
        end = start + 20  # 개수 20개
        fig, axs = plt.subplots(2, (end - start + 1) // 2, figsize=(20, 5))
        fig.subplots_adjust(hspace=.01, wspace=.01)
        axs = axs.ravel()

        for i in range(start, end):
            cam_img, original_img, label, predict_label = self.cam(trainset, i, img_size)

            axs[i - start].imshow(original_img, cmap='gray')
            axs[i - start].imshow(cam_img, cmap='jet', alpha=.3)  # cmap - histogram? 색상 , alpha 투명도
            axs[i - start].set_title(f'lbl : {label}\nPre : {predict_label}')
            axs[i - start].axis('off')
        fig.suptitle(f'epoch - {epoch}')

        # plt.show()
        fig.savefig(f'{self.save_path}/cam_{epoch}.png')
