import json
import os
import pandas as pd

# import openpyxl

"""
    faster r cnn ouput to json file and compare with label json file
"""

file_path = r"C:\Object_Detection\data\remake\instatnce-tooth\sample_data_json"
file_name = 'sample_data.json'
coco_json_path = r'C:\Object_Detection\data\remake\instatnce-tooth\annotations\instances_test2017.json'
if file_name not in os.listdir(file_path):
    file = open(file_path + "/" + file_name, 'w', encoding='utf-8')


def predict_to_json(model_output_dict):
    with open(file_path + '/' + file_name, 'w', encoding='utf-8') as file:
        json.dump(model_output_dict, file, ensure_ascii=False, indent='\t')


def remake_coco_json(coco_json_path):
    with open(coco_json_path, 'r') as file:
        data = json.load(file)

    id = {}
    category = {}

    for j in data["images"]:
        id[j['id']] = j['file_name']

    for k in data['categories']:
        if k['id'] == 0:
            continue
        category[k['id']] = str(int(k['name']) + 1)

    new_dict = {}
    for i in data["annotations"]:
        if id[i['image_id']] in new_dict:
            pass
        else:
            new_dict[id[i['image_id']]] = {}

        new_dict[id[i['image_id']]][category[i['category_id']]] = [i['bbox']]
        bbox_list = new_dict[id[i['image_id']]][category[i['category_id']]][0]
        bbox_list[2] = bbox_list[0] + bbox_list[2]
        bbox_list[3] = bbox_list[1] + bbox_list[3]
        new_dict[id[i['image_id']]][category[i['category_id']]] = [bbox_list]

    with open(file_path + '/remake_' + file_name, 'w', encoding='utf-8') as file:
        json.dump(new_dict, file, ensure_ascii=False, indent='\t')


def make_iou(predict_json, label_json):
    iou_dict = {}

    target = {'1': 'RU1', '2': 'RU2', '3': 'LU3', '4': 'LU4', '5': 'LU5', '6': 'LU6', '7': 'LU7', '8': 'LU8', '9': 'LL1',
                          '10': 'LL2', '11': 'LL3', '12': 'LL4', '13': 'RU3', '14': 'LL5', '15': 'LL6', '16': 'LL7', '17': 'LL8', '18': 'RL1',
                          '19': 'RL2', '20': 'RL3', '21': 'RL4', '22': 'RL5', '23': 'RL6', '24': 'RU4', '25': 'RL7', '26': 'RL8', '27': 'RU5', '28': 'RU6',
                          '29': 'RU7', '30': 'RU8', '31': 'LU1', '32': 'LU2'}
    all_tooth = []
    for i in range(1, 32):
        all_tooth.append(str(i))

    wrong_dict = {}
    with open(predict_json, 'r') as p:
        predict_dict = json.load(p)

    with open(label_json, 'r') as l:
        label_dict = json.load(l)

    for i in list(predict_dict.keys()):

        iou_dict[i] = {}
        for j in range(32):
            j += 1
            if str(j) not in predict_dict[i]:
                continue

            try:
                list_predict = []
                if len(predict_dict[i][str(j)]) != 1:
                    for m in predict_dict[i][str(j)]:
                        list_predict.append(m[4])
                    index = list_predict.index(max(list_predict))
                    predict_box = predict_dict[i][str(j)][index]
                else:
                    predict_box = predict_dict[i][str(j)][0]
                label_box = label_dict[i][str(j)][0]
            except:
                continue

            xA = max(predict_box[0], label_box[0])
            yA = max(predict_box[1], label_box[1])
            xB = min(predict_box[2], label_box[2])
            yB = min(predict_box[3], label_box[3])
            intersection_area = (xB - xA) * (yB - yA)
            # Calculate the union area

            boxA_area = (predict_box[2] - predict_box[0]) * (predict_box[3] - predict_box[1])
            boxB_area = (label_box[2] - label_box[0]) * (label_box[3] - label_box[1])
            union_area = boxA_area + boxB_area - intersection_area

            # Calculate the IoU
            iou = intersection_area / union_area
            iou_dict[i][str(j)] = iou

        predict_key_list = list(predict_dict[i].keys())
        label_key_list = list(label_dict[i].keys())
        missing = list(set(all_tooth) - set(label_key_list))    # label missing 치아
        missing_detect = list(set(all_tooth)-set(predict_key_list)) # detect 하지 못한 치아

        TF = list(set(label_key_list) - set(predict_key_list))      # TF - label 존재, predict 존재 x
        TT = list(set(label_key_list) & set(predict_key_list))      # TT - label 존재, predict 존재
        FT = list(set(missing) & set(predict_key_list))             # FT - label 존재 x, predict 존재
        FF = list(set(missing) & set(missing_detect))               # FF - label 존재 x, predict 존재 x
        print(missing, predict_key_list)
        try:
            for k in range(len(FF)):
                FF[k] = target[FF[k]]
        except:
            pass
        try:
            for k in range(len(TT)):
                TT[k] = target[TT[k]]
        except:
            pass
        try:
            for k in range(len(FT)):
                FT[k] = target[FT[k]]
        except:
            pass

        try:
            for k in range(len(TF)):
                TF[k] = target[TF[k]]
        except:
            pass



        iou_dict[i]['_FT'] = len(FT)
        iou_dict[i]['_FT_list'] = sorted(FT)
        iou_dict[i]['_TF'] = len(TF)
        iou_dict[i]['_TF_list'] = sorted(TF)
        iou_dict[i]['_TT'] = len(TT)
        iou_dict[i]['_TT_list'] = sorted(TT)
        iou_dict[i]['_FF'] = len(FF)
        iou_dict[i]['_FF_list'] = sorted(FF)
        # iou_dict[i]['missing'] = len(missing)
        # iou_dict[i]['missing_list'] = missing
        # iou_dict[i]['missing_detect'] = len(missing_detect)
        # iou_dict[i]['missing_detect_list'] = missing_detect

    # print(iou_dict)
    return iou_dict


def dict_to_xlsx(dict):
    new_dict = {}
    name = []
    # print(list(dict.keys()),list(dict.values()))
    for i in list(dict.keys()):
        name.append(i.split('.')[0])

    for i in range(len(name)):
        new_dict[name[i]] = list(dict.values())[i]

    df = pd.DataFrame.from_dict(data=new_dict)
    df = df.rename(index={'1': 'RU1', '2': 'RU2', '3': 'LU3', '4': 'LU4', '5': 'LU5', '6': 'LU6', '7': 'LU7', '8': 'LU8', '9': 'LL1',
                          '10': 'LL2', '11': 'LL3', '12': 'LL4', '13': 'RU3', '14': 'LL5', '15': 'LL6', '16': 'LL7', '17': 'LL8', '18': 'RL1',
                          '19': 'RL2', '20': 'RL3', '21': 'RL4', '22': 'RL5', '23': 'RL6', '24': 'RU4', '25': 'RL7', '26': 'RL8', '27': 'RU5', '28': 'RU6',
                          '29': 'RU7', '30': 'RU8', '31': 'LU1', '32': 'LU2'})
    df.sort_index(axis=0, sort_remaining=True).to_excel(excel_writer=r'C:\Object_Detection\data\remake\instatnce-tooth/' + 'test1.xlsx')


if __name__ == "__main__":
    remake_coco_json(coco_json_path)
    iou_dict = make_iou(r"C:\Object_Detection\data\remake\instatnce-tooth\sample_data_json\sample_data.json",
                        r"C:\Object_Detection\data\remake\instatnce-tooth\sample_data_json\remake_sample_data.json")
    dict_to_xlsx(iou_dict)
