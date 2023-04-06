import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F
import torch.nn as nn


class V_Temlplate_VGG11_norm(nn.Module):
    def __init__(self, in_channels: int =1, num_classes: int=1 ):

        super(V_Temlplate_VGG11_norm, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv3d(self.in_channels, 64, kernel_size=3, padding=1)
        self.BN3d_1 = nn.BatchNorm3d(64)
        # nn.ReLU(),
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.BN3d_2 = nn.BatchNorm3d(128)
        # nn.ReLU(),
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.BN3d_3 = nn.BatchNorm3d(256)
        # nn.ReLU(),
        self.conv4 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.BN3d_4 = nn.BatchNorm3d(256)
        # nn.ReLU(),
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.BN3d_5 = nn.BatchNorm3d(512)
        # nn.ReLU(),
        self.conv6 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.BN3d_6 = nn.BatchNorm3d(512)
        # nn.ReLU(),
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv7 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.BN3d_7 = nn.BatchNorm3d(512)
        # nn.ReLU(),
        self.conv8 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.BN3d_8 = nn.BatchNorm3d(512)
        # nn.ReLU(),
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        # 여기 if 문들에서는 각각 템플릿타입에 따라 input_사이즈에 따라 중간출력물의 사이즈가 달라지므로 템플릿타입에 따른
        # 통과할 망의 크기를 정의해 놓은 것이다.
        self.Linear1 = nn.Linear(in_features=1024 * 4 * 1 * 1, out_features=4096) #1536
        # else:
        #
        #     logging.debug("input_size를 256*32*32 로 맞추십시오")
        #     exit()

        self.drop = nn.Dropout3d(0.5)
        self.Linear2 = nn.Linear(in_features=4096, out_features=4096)
        # nn.ReLU(),
        self.drop = nn.Dropout3d(0.5)
        self.Linear3 = nn.Linear(in_features=4096, out_features=self.num_classes)


    def forward(self, x: torch.Tensor):

        # input_size = x.shape
        x = self.conv1(x)
        x = F.relu(self.BN3d_1(x))
        x = self.maxpool(x)
        x = self.conv2(x)
        x = F.relu(self.BN3d_2(x))
        x = self.maxpool(x)
        x = self.conv3(x)
        x = F.relu(self.BN3d_3(x))
        x = self.conv4(x)
        x = F.relu(self.BN3d_4(x))
        x = self.maxpool(x)
        x = self.conv5(x)
        x = F.relu(self.BN3d_5(x))
        x = self.conv6(x)
        x = F.relu(self.BN3d_6(x))
        x = self.maxpool(x)
        x = self.conv7(x)
        x = F.relu(self.BN3d_7(x))
        x = self.conv8(x)
        x = F.relu(self.BN3d_8(x))
        x = self.maxpool(x)
        # print("x1_l_cnn:",x.size())
        # print("x.shape:", x.shape)
        x = x.view(x.size(0), -1)
        # print("x1_L:",x.size())
        x = self.Linear1(x)
        x = self.drop(x)
        x = self.Linear2(x)
        x = self.drop(x)
        x = self.Linear3(x)

        return x
###################################################################

class FeatureMapVisualizer:
    def __init__(self, model, image_size=(224, 224), num_maps=20):  # image 사이즈 확인, num_maps = feature map 을 보여 주는 개수

        self.model = model
        self.image_size = image_size
        self.num_maps = num_maps
        self.feature_maps = {}

    def _get_feature_maps_hook(self, name):
        """hook 이라는 함수를 사용하여 입력한 layer 이름의 정보들을 저장 하는 역할? 자세하게는 모름"""

        def hook(model, input, output):
            self.feature_maps[name] = output.detach()

        return hook

    def visualize(self, image, layer_name_feature_maps_number: dict):
        print(123)
        """
            image - dataset 에서 정규화를 거쳐 input에 들어가는 형식 넣기
            layer_name_feature_maps_number - {'features.0': [0, 1, 2, 3, 4], 'features.1': [0, 1, 2, 3]}
        """
        # dict - key=layer 명, value=feature_map 번호
        if not layer_name_feature_maps_number:  # 없으면 전체
            for name, layer in self.model.named_modules():
                if isinstance(layer, nn.Conv3d):
                    layer.register_forward_hook(self._get_feature_maps_hook(name))

        else:
            for i in layer_name_feature_maps_number.keys():
                for name, layer in self.model.named_modules():
                    if isinstance(layer, nn.Conv3d) and i == name:
                        layer.register_forward_hook(self._get_feature_maps_hook(i))
                        print(i, name)

        output = self.model(image)  # ??
        for name, maps in self.feature_maps.items():  # hook 에서 저장한 layer 가져옴
            print(f"Layer: {name}")
            print(f"Shape: {maps.shape}")
            num_maps = maps.shape[1]

            # dict 에 입력한 feature map 번호를 저장
            if not layer_name_feature_maps_number:
                selected_maps = random.sample(range(num_maps), min(num_maps, self.num_maps))
            else:
                selected_maps = layer_name_feature_maps_number[name]

            print(selected_maps)

            # 해당되는 layer, feature map 출력
            fig = plt.figure(figsize=(15, 5))

            # fig.subplots_adjust(wspace=0.1, hspace=0.4, top=0.9, bottom=0.1, left=0.05, right=0.95)
            row = len(selected_maps) // 10 + 1
            if len(selected_maps) < 10:
                column = len(selected_maps)
            else:
                column = 10

            for i, j in enumerate(selected_maps):
                ax = fig.add_subplot(row, column, i + 1, projection='3d')
                X, Y = np.meshgrid(np.arange(maps.shape[-2]), np.arange(maps.shape[-3]))
                Z = maps[0, 0, j, :, :]
                ax.plot_surface(X, Y, Z)
                ax.axis('off')  # grid 제거?
                ax.set_title(f"Map {j}")  # subplot name
            fig.suptitle(name)
            plt.show()

    def print_model_info(self):
        """모델 정보 불러오기 그냥 for문 사용하면 됨"""
        for name in self.model.named_modules():
            print(name)


if __name__ == "__main__":
    # model = models.vgg16(pretrained=True)
    model = V_Temlplate_VGG11_norm()

    visualizer = FeatureMapVisualizer(model)  # model 넣기
    image1 = torch.randn(1, 1, 64, 64, 64)
    print(image1)
    # image = test_dataset[1][0].unsqueeze(0) # sample
    visualizer.visualize(image1, {'conv1': [1, 2, 3, 4, 5], 'conv3': [1, 2, 3, 4, 5], 'conv5': [1, 2, 3, 4, 5]})
