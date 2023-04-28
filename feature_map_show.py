import os

import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import random

import torchvision.transforms as transforms
from torch.utils.data import DataLoader  # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리

'''
    feature map 을 뽑아내는 코드
    
    FeatureMapVisualizer(model, path, mk_epoch, use=True)
    model - model 입력
    path - feature map 을 저장할 경로
    mk_epoch - 몇 배수마다 epoch 안에서 생성할지 결정
    use = True 사용 여부, False 로 하면 생성되지 않음
    
    visualize(epoch, image, input_name, layer_name_feature_maps_number):
    epoch - 전체 epoch
    image - dataloder 에서 들어가는 image
    input_name - dataloader 에서 들어가는 image name
    layer_name_feature_maps_number - None 일때 첫번째, 마지막 layer, 첫번째 마지막 feature map + random feature map 개수
                                   - dict 형식으로 입력하면 해당되는 feature map 출력 ex) {'features.0': [0, 1, 2, 3, 4], 'features.1': [0, 1, 2, 3]}
                                   
    train 시작지점 에서 아래의 형식의 코드 입력

    for epoch in range(self.epoch_size):
        visualizer = FeatureMapVisualizer(self.model, feature_map_save_path, feature_map_save_epoch, use=feature_map)   # feature map 생성 선언
        visualizer.create_feature_map_epoch_folder(epoch)  # feature map 폴더 안 epoch 폴더 생성
    
    trainloder 안에 아래 형식으로 입력
    
    #   dataloader 는 custom dataset 으로 세팅 하였으며 data['filename'] 은 custom data set 안에서 정의함,
        dataset을 정의 하지 않을경우 data['filename'] 은 임시로 넣으면됨 - data type은 확인 하지 않았으나 list 로 추정 

    for i, data in trainloader:
        inputs, values = data['image'].to(self.device), data['label'].to(self.device)   
        self.optimizer.zero_grad()  # 최적화 초기화

                        ***
                        
        if i == 1:  # 첫번째 trainlodar 에서만 입력
            visualizer.visualize(epoch, inputs, data['filename'], feature_map_layer_name)  # feature_map - epoch 폴더 안에 생성
'''


class FeatureMapVisualizer:
    feature_map_folder_name = 'feature_map'

    def __init__(self, model, path: str, mk_epoch: int, image_size=(224, 224), num_maps=5, use=True):  # image 사이즈 확인, num_maps = feature map 을 보여 주는 개수

        self.model = model
        self.path = path + '/' + FeatureMapVisualizer.feature_map_folder_name
        self.image_size = image_size
        self.num_maps = num_maps
        self.feature_maps = {}
        self.use = use
        self.mk_epoch = mk_epoch

        if self.use and FeatureMapVisualizer.feature_map_folder_name not in os.listdir(path):  # 사용 상태이고 path 에 해당 폴더 없으면 생성
            os.mkdir(self.path)  # feature_map 저장할 폴더 생성

    def _get_feature_maps_hook(self, name):
        """hook 이라는 함수를 사용하여 입력한 layer 이름의 정보들을 저장 하는 역할? 자세하게는 모름"""

        def hook(model, input, output):
            self.feature_maps[name] = output.detach()

        return hook

    def visualize(self, epoch: int, image, input_name, layer_name_feature_maps_number: dict):
        """
            image - dataset 에서 정규화를 거쳐 input에 들어가는 형식 넣기
            layer_name_feature_maps_number - {'features.0': [0, 1, 2, 3, 4], 'features.1': [0, 1, 2, 3]}
        """
        if self.use and (epoch + 1) % self.mk_epoch == 0:  # self.use == True 이고, 설정한 epoch 배수 마다 설정

            if not layer_name_feature_maps_number:  # 지정한 feature map 없을 때 전체에서 첫번째, 마지막 layer 만 hook 하기 위함
                save_name = []
                i = 0
                for name, layer in self.model.named_modules():  # module 정보
                    i += 1
                    if isinstance(layer, nn.Conv2d):  # layer 와 conv2 에 해당되는 것만 가지고옴
                        save_name.append(name)  # layer 이름 저장
                for name, layer in self.model.named_modules():
                    if isinstance(layer, nn.Conv2d) and name == save_name[0] or name == save_name[-1] or name == save_name[len(save_name)//2]:  # 저장한 layer 중 첫번 째 마지막 layer 만 hook
                        layer.register_forward_hook(self._get_feature_maps_hook(name))

            else:
                for i in layer_name_feature_maps_number.keys():
                    for name, layer in self.model.named_modules():
                        if isinstance(layer, nn.Conv2d) and i == name:
                            layer.register_forward_hook(self._get_feature_maps_hook(i))

            output = self.model(image)  # ??

            for name, maps in self.feature_maps.items():  # hook 에서 저장한 layer 가져옴
                print(f"Layer: {name}")
                print(f"Shape: {maps.shape}")
                print(maps.shape[1])
                num_maps = maps.shape[1]

                # dict 에 입력한 feature map 번호를 저장
                if not layer_name_feature_maps_number:
                    selected_maps = random.sample(range(num_maps), min(num_maps + 1, self.num_maps - 1))  # 랜덤으로 입력값 지정
                    selected_maps.append(0)  # index 0 번째 feature map
                    selected_maps.append(num_maps - 1)  # 마지막 index feature map
                    # selected_maps.append()
                else:
                    selected_maps = layer_name_feature_maps_number[name]
                selected_maps.sort()

                # print(selected_maps)

                # 해당되는 layer, feature map 출력
                fig = plt.figure(figsize=(10, 5))
                # fig.subplots_adjust(wspace=0.1, hspace=0.4, top=0.9, bottom=0.1, left=0.05, right=0.95)
                row = len(selected_maps) // 10 + 1

                if len(selected_maps) > 9:  # 10 이하의 이미지 개수에 맞춰서 column 의 길이 조정 하기 위해 사용
                    column_size = 10
                else:
                    column_size = len(selected_maps)

                for i, j in enumerate(selected_maps):
                    ax = fig.add_subplot(row, column_size, i + 1)  # row,column,index =>ex) 3,3,2 => 3x3 의 2번째 index
                    ax.imshow(maps[0, j, :, :].cpu().numpy(), cmap='gray')  # ?? cmap='gray' 지우면 색상 나옴
                    ax.axis('off')  # grid 제거
                    ax.set_title(f"Map {j}", fontsize=7)  # subplot name

                # print(input_name)
                # if len(input_name) != 1:  # batch size 가 1 이상일때 아닐때 title 명 조정
                #     fig.suptitle(f'{name}\n{input_name[0]}')
                # else:
                #     fig.suptitle(f'{name}\n{input_name[-1]}')

                plt.savefig(f'{self.path}/{epoch + 1}/{name}.png')

                # plt.show()    # plt.save 랑 같이 사용 불가능 / show 사용시 모든 os.mkdir 제거하고 하면 편함 ( 안해도 됨 )

    def print_model_info(self):
        """모델 정보 불러오기 그냥 for문 사용하면 됨"""
        for name in self.model.named_modules():
            print(name)

    def create_feature_map_epoch_folder(self, i):
        if (i + 1) % self.mk_epoch == 0 and self.use == True:
            os.mkdir(f'{self.path}/{i + 1}')  # 저장할 폴더 생성


if __name__ == "__main__":
    feature_map_save_path = r'C:\woo_project\AI_Study\sample_data'
    epoch = 10
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48827466, 0.4551035, 0.41741803), (0.05540232, 0.054113153, 0.055464733))
    ])  # 데이터 정규화
    # custom = test.CustomDataset(r'C:\woo_project\AI_Study\sample_data\sample', transform_train)
    # dataloader = torch.utils.data.DataLoader(custom, 1, shuffle=True)
    model = models.vgg16(pretrained=True)
    visualizer = FeatureMapVisualizer(model, feature_map_save_path, 2, use=True)  # model 넣기
    image1 = torch.randn(1, 3, 224, 224)
    # image = test_dataset[1][0].unsqueeze(0) # sample
    print(model)

    os.mkdir(feature_map_save_path + FeatureMapVisualizer.feature_map_folder_name)  # feature_map 저장할 폴더 생성

    for i in range(epoch):
        visualizer.create_feature_map_epoch_folder(i)
        for j, b in enumerate(dataloader):
            inputs = b['image'].to('cpu')
            labels = b['label'].to('cpu')
            names = b['filename']
            visualizer.visualize(i, inputs, names, None)
