import os

import torch
import torchvision
import argparse
import cv2
import sys
import coco_names

sys.path.append('./')
import random
import test
import tqdm

"""
    predict 결과에서 txt 또는 json 파일로 저장한 뒤 기존 json 파일과 iou 를 계산할 필요가 있음.
"""


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')

    parser.add_argument('--model_path', type=str, default=r'D:\Object Detection\instance_tooth_save_result/model_1.pth', help='model path')
    parser.add_argument('--image_path', type=str,
                        default=r'C:\Object_Detection\data\remake\instatnce-tooth\test2017', help='image path')  # 이미지 폴더 경로
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='model')
    parser.add_argument('--score', type=float, default=0.75, help='objectness score threshold')
    # objectness score = box confidence score * conditional class probabilty
    args = parser.parse_args()

    return args


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)


def main():
    args = get_args()
    if args.dataset == 'coco':
        num_classes = 33
        names = coco_names.names
    # Model creating
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=False)
    model = model.cpu()  # 회사 gpu 로는 동작하지 않아서 cpu 사용
    model.eval()

    save = torch.load(args.model_path)
    model.load_state_dict(save['model'])

    list_image = os.listdir(args.image_path)

    to_json = {}

    for j in list_image:
        input = []
        image = args.image_path + '/' + j
        src_img = cv2.imread(image)
        img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cpu()  # 회사 gpu 로는 동작하지 않아서 cpu 사용
        input.append(img_tensor)
        out = model(input)

        boxes = out[0]['boxes']
        labels = out[0]['labels']
        scores = out[0]['scores']
        to_json[j] = {}

        for idx in range(boxes.shape[0]):
            if scores[idx] >= args.score:
                x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
                name = names.get(str(labels[idx].item()))
                try:
                    to_json[j][str(labels[idx].item())].append([x1, y1, x2, y2, float(scores[idx])])
                except:
                    to_json[j][str(labels[idx].item())] = []
                    to_json[j][str(labels[idx].item())].append([x1, y1, x2, y2, float(scores[idx])])

                # cv2.rectangle(img,(x1,y1),(x2,y2),colors[labels[idx].item()],thickness=2) # 클래스별로 색상 지정 가능
                cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
                cv2.putText(src_img, text=name + ':' + str(round(float(scores[idx]), 4)), org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

        # cv2.imshow('result', src_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        cv2.imwrite(fr'C:\Object_Detection\data\remake\instatnce-tooth\sample_result/{j}', src_img)
    test.predict_to_json(to_json)


if __name__ == "__main__":
    main()
