"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from tqdm.autonotebook import tqdm
import torch
from pycocotools.cocoeval import COCOeval
import feature_map_show


# from apex import amp


def train(model, train_loader, epoch, writer, criterion, optimizer, scheduler, is_amp):
    model.train()
    feature_map_save_path = r'C:\새 폴더'
    visualizer = feature_map_show.FeatureMapVisualizer(model, feature_map_save_path, 1, use=True)  # feature map 생성 선언
    visualizer.create_feature_map_epoch_folder(epoch)  # feature map 폴더 안 epoch 폴더 생성
    num_iter_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader)
    for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
        if torch.cuda.is_available():
            img = img.cuda()
            gloc = gloc.cuda()
            glabel = glabel.cuda()
        if i == 1:  # 첫번째 trainlodar 에서만 입력
            visualizer.visualize(epoch=epoch, image=img, input_name=None, layer_name_feature_maps_number=None)  # feature_map - epoch 폴더 안에 생성
        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()
        loss = criterion(ploc, plabel, gloc, glabel)

        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))

        writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)

        # if is_amp:
        #     with amp.scale_loss(loss, optimizer) as scale_loss:
        #         scale_loss.backward()
        # else:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()


def evaluate(model, test_loader, epoch, writer, encoder, nms_threshold):
    model.eval()
    detections = []
    category_ids = test_loader.dataset.coco.getCatIds()
    for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):
        print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            # Get predictions
            ploc, plabel = model(img)
            ploc, plabel = ploc.float(), plabel.float()

            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]  # mns threshold
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue

                height, width = img_size[idx]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    detections.append([img_id[idx], loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,
                                       (loc_[3] - loc_[1]) * height, prob_,
                                       category_ids[label_ - 1]])

    detections = np.array(detections, dtype=np.float32)

    coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)
