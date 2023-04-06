import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import random


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
        """
            image - dataset 에서 정규화를 거쳐 input에 들어가는 형식 넣기
            layer_name_feature_maps_number - {'features.0': [0, 1, 2, 3, 4], 'features.1': [0, 1, 2, 3]}
        """
        # dict - key=layer 명, value=feature_map 번호
        if not layer_name_feature_maps_number:  # 없으면 전체
            for name, layer in self.model.named_modules():

                if isinstance(layer, nn.Conv2d):
                    layer.register_forward_hook(self._get_feature_maps_hook(name))
                    # print(layer)
        else:
            for i in layer_name_feature_maps_number.keys():
                for name, layer in self.model.named_modules():
                    if isinstance(layer, nn.Conv2d) and i == name:
                        layer.register_forward_hook(self._get_feature_maps_hook(i))
                        print(i, name)

        output = self.model(image)  # ??

        for name, maps in self.feature_maps.items():  # hook 에서 저장한 layer 가져옴
            print(f"Layer: {name}")
            print(f"Shape: {maps.shape}")
            print(maps.shape)
            num_maps = maps.shape[1]

            # dict 에 입력한 feature map 번호를 저장
            if not layer_name_feature_maps_number:
                selected_maps = random.sample(range(num_maps), min(num_maps, self.num_maps))
            else:
                selected_maps = layer_name_feature_maps_number[name]

            print(selected_maps)

            # 해당되는 layer, feature map 출력
            fig = plt.figure(figsize=(10, 5))
            # fig.subplots_adjust(wspace=0.1, hspace=0.4, top=0.9, bottom=0.1, left=0.05, right=0.95)
            row = len(selected_maps) // 10 + 1

            for i, j in enumerate(selected_maps):
                ax = fig.add_subplot(row, 10, i + 1)  # row,column,index =>ex) 3,3,2 => 3x3 의 2번째 index
                ax.imshow(maps[0, j, :, :].cpu().numpy())  # ?? cmap='gray' 지우면 색상정보 전부 나옴
                ax.axis('off')  # grid 제거?
                ax.set_title(f"Map {j}")  # subplot name
            fig.suptitle(name)
            plt.show()

    def print_model_info(self):
        """모델 정보 불러오기 그냥 for문 사용하면 됨"""
        for name in self.model.named_modules():
            print(name)


if __name__ == "__main__":
    model = models.vgg16(pretrained=True)
    visualizer = FeatureMapVisualizer(model)  # model 넣기
    image1 = torch.randn(1, 3, 224, 224)
    # image = test_dataset[1][0].unsqueeze(0) # sample
    visualizer.visualize(image1, {'features.0': [0, 1, 2, 3, 4], 'features.1': [0, 1, 2, 3]})
