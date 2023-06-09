import utils
import dataset.transforms as T
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
from dataset.coco_utils import get_coco, get_coco_kp
from engine import train_one_epoch, evaluate
from dataset.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
import argparse
import torchvision
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import grad_cam
import cv2
import random


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Training')

    parser.add_argument('--data_path', default=r'C:\woo_project\AI_Study\object_detection\data_\coco', help='dataset path') # train val test 상위 경로
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='dataset')  # coco format 강의에서 사용한 포멧
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--b', '--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=200, type=int, help='print frequency')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[300, 450], nargs='+', type=int, help='decrease lr every step-size epochs')   # learning rate 언제 줄일지
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')  # 얼마의 비율로 learning rate 줄일지
    parser.add_argument('--resume', default=r'D:\Object Detection\instance_tooth_save_result\model_augmentation.pth', help='resume from checkpoint')     # pth 파일 입력
    parser.add_argument('--test_only', default=False, type=bool, help='resume from checkpoint')     # evaluate 만 실행 할 경우 입력
    parser.add_argument('--output-dir', default=r'D:\Object Detection\instance_tooth_save_result', help='path where to save')
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--distributed', default=True, help='if distribute or not')  # (분산 학습) 기법중하나 정확한 방식 모름
    parser.add_argument('--parallel', default=False, help='if distribute or not')
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')  # gpu가 1개 인 상황으로 알고있음
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


def get_dataset(name, image_set, transform):
    paths = {
        "coco": (r'C:\Object_Detection\data\remake\instatnce-tooth_augmentation2', get_coco, 33),  # 제일 뒤에는 class 개수
        "coco_kp": ('/datasets01/COCO/022719/', get_coco_kp, 2)  # custom data 를 생성하면 제작
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # tensorboard 제작
    log_dir = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('tensorboard/log', log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    args = get_args()
    if args.output_dir:
        utils.mkdir(args.output_dir)
    utils.init_distributed_mode(args)

    # Data loading
    print("Loading data")
    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True))
    dataset_val, _ = get_dataset(args.dataset, "val", get_transform(train=False))
    dataset_test, _ = get_dataset(args.dataset, "test", get_transform(train=False))     # test dataset 직접 제작

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)

    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.b)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.b, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.b,
        sampler=val_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.b,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    # Model creating
    print("Creating model")
    # model = models.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained)   
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                              pretrained=False, pretrained_backbone=False)

    device = torch.device(args.device)
    model.to(device)
    # print(model)

    # Distribute
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

        # Parallel
    if args.parallel:
        print('Training parallel')
        model = torch.nn.DataParallel(model).cuda()
        model_without_ddp = model.module

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # Resume training
    if args.resume:
        print('Resume training')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        model.eval()
        evaluate(model, data_loader_test, device=device,epoch=0, writer=writer)
        conv2d_layers = []
        # for name, module in model.named_modules():
        #     if isinstance(module, nn.Conv2d):
        #         conv2d_layers.append(name)
        #
        # grad_cam.insert_input_module_layer(model, 0, conv2d_layers, dataset_test, image=None)
        # return

    # Training
    print('Start training')
    start_time = time.time()
    start_loss = 100


    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, writer)

        # fixme train 기준으로 현재 파일이 저장되고 있음, valid 기존으로 되도록 코드 변경 필요함,,.
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(args.output_dir, f'model_{epoch}.pth'))

        # evaluate after every epoch
        evaluate(model, data_loader_val, device=device, epoch=epoch, writer=writer)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
