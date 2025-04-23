from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, cohen_kappa_score, recall_score, f1_score
import numpy as np
import pandas as pd

def compute_auc(cls_scores, cls_labels):
    cls_aucs = []
    for i in range(cls_scores.shape[1]):
        scores_per_class = cls_scores[:, i]
        labels_per_class = cls_labels[:, i]
        auc_per_class = roc_auc_score(labels_per_class, scores_per_class)
        cls_aucs.append(auc_per_class)

    return cls_aucs

def compute_metrics(gt_label, pred_label, pred_scores, test_tumor_id):
    test_accuracy = accuracy_score(gt_label, pred_label)
    test_report = classification_report(gt_label, pred_label, digits=3)
    test_kappa = cohen_kappa_score(gt_label, pred_label)
    cm = confusion_matrix(gt_label, pred_label)

    specificity = []
    n_classes = cm.shape[0]
    for i in range(n_classes):
        true_negatives = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        false_positives = np.sum(cm[:, i]) - cm[i, i]
        specificity_i = true_negatives / (true_negatives + false_positives)
        specificity.append(round(specificity_i,3))
    print("specificity:", specificity)

    sensitivity = recall_score(gt_label, pred_label, average=None).round(3)
    f1_scores = f1_score(gt_label, pred_label, average=None).round(3)

    print('test accuracy: {:.3f}'.format(test_accuracy))
    print('test kappa: {:.3f}'.format(test_kappa))
    print("\nClassification Report:\n", test_report)

    gt_label_onehot = np.zeros(pred_scores.shape)
    for k in range(pred_scores.shape[0]):
        label = gt_label[k]
        one_hot_label = np.array([int(i == label) for i in range(pred_scores.shape[1])])
        gt_label_onehot[k, :] = one_hot_label

    cls_aucs = compute_auc(pred_scores, gt_label_onehot)
    cls_aucs = np.round(cls_aucs, 3)

    # Collect individual sample metrics
    sample_metrics = []
    for i in range(len(gt_label)):
        sample_data = {
            "image_id": f"{test_tumor_id[i]}",  # Replace this with actual image ID if available
            "label": int(gt_label[i]),
            "prediction": int(pred_label[i]),
            "score": pred_scores[i].tolist()
        }
        sample_metrics.append(sample_data)
    return test_accuracy,test_kappa, specificity, sensitivity, f1_scores, cls_aucs, sample_metrics
