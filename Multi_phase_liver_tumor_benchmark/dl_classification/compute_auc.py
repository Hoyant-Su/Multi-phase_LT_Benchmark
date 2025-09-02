import numpy as np
from sklearn import metrics
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

### 1. compute auc for each class
def compute_auc(cls_scores, cls_labels):
    cls_aucs = []
    for i in range(cls_scores.shape[1]):
        scores_per_class = cls_scores[:, i]
        labels_per_class = cls_labels[:, i]
        auc_per_class = metrics.roc_auc_score(labels_per_class, scores_per_class)   
        cls_aucs.append(auc_per_class)
    
    return cls_aucs

def load_score(score_json):
    with open(score_json, 'r') as f:
        score_dict = json.load(f)
    return score_dict
