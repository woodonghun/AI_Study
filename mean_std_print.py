import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets


def print_stats(dataset):
    imgs = np.array([img.numpy() for img, _ in dataset])
    print(f'shape: {imgs.shape}')

    min_r = np.min(imgs, axis=(2, 3))[:, 0].min()
    min_g = np.min(imgs, axis=(2, 3))[:, 1].min()
    min_b = np.min(imgs, axis=(2, 3))[:, 2].min()

    max_r = np.max(imgs, axis=(2, 3))[:, 0].max()
    max_g = np.max(imgs, axis=(2, 3))[:, 1].max()
    max_b = np.max(imgs, axis=(2, 3))[:, 2].max()

    mean_r = np.mean(imgs, axis=(2, 3))[:, 0].mean()
    mean_g = np.mean(imgs, axis=(2, 3))[:, 1].mean()
    mean_b = np.mean(imgs, axis=(2, 3))[:, 2].mean()

    std_r = np.std(imgs, axis=(2, 3))[:, 0].std()
    std_g = np.std(imgs, axis=(2, 3))[:, 1].std()
    std_b = np.std(imgs, axis=(2, 3))[:, 2].std()

    print(f'min: {min_r, min_g, min_b}')
    print(f'max: {max_r, max_g, max_b}')
    print(f'mean: {mean_r, mean_g, mean_b}')
    print(f'std: {std_r, std_g, std_b}')

    return mean_r, mean_g, mean_b, std_r, std_g, std_b


if __name__ == "__main__":
    data_path_train = r'D:\AI_study\cnn\3. 8종 이미지 분류\train_natural_images'
    data_path_test = r'D:\AI_study\cnn\3. 8종 이미지 분류\test_natural_images'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])  # 데이터 정규화
    train_dataset = torchvision.datasets.ImageFolder(root=data_path_train, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=data_path_test, transform=transform)
    print_stats(train_dataset)
    print_stats(test_dataset)
