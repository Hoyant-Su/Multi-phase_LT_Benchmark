import os
import json
from sklearn import metrics

def load_pred_label(json_file):
    gt_list = []
    pred_list = []
    with open(json_file, 'r') as f:
        load_info = json.load(f)
        for item in load_info:
            pred = item['prediction']
            label = item['label']
            print (item['image_id'], pred, label)
            
            gt_list.append(label)
            pred_list.append(pred)

    return gt_list, pred_list
