from joblib import dump
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from Multi_phase_liver_tumor_benchmark.radiomics_extract_classification.src.feature_importance import extract_feature_importance, plot_feature_importance, rank_feature_importance
from Multi_phase_liver_tumor_benchmark.radiomics_extract_classification.src.feature_selection import remove_correlated_features
from Multi_phase_liver_tumor_benchmark.radiomics_extract_classification.src.model_evaluation import adjusted_prediction, eval_sensitivity_specificity
from Multi_phase_liver_tumor_benchmark.radiomics_extract_classification.phase_feat_concat import radiomics_data_parse

import os
import csv
import globals

from imblearn.over_sampling import SMOTE

# Feature selection
#corr_threshold = 0.8
corr_threshold = globals.corr_threshold

# Fix seed
seed = 42
np.random.seed(seed)
random.seed(seed)

do_oversample = 'False'

exp = globals.exp
date = globals.date
fold_name = globals.fold_name
do_extension = globals.do_extension


file_root_path = f'../../tumor_radiomics/Label/exp/{exp}/{fold_name}'  
os.makedirs(file_root_path,exist_ok=True)
phases_num = globals.phases_num


if __name__ == '__main__':
    fold_num = 5
    for fold_index in range(1, fold_num+1):
    
        feature_file_training_set_art = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_art.csv')
        feature_file_training_set_delay = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_delay.csv')
        feature_file_training_set_nc = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_nc.csv')
        feature_file_training_set_pvp = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_pvp.csv')

        feature_file_training_set_dilated_ring_pvp = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_pvp_dilated_ring.csv')
        feature_file_training_set_dilated_ring_art = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_art_dilated_ring.csv')
        feature_file_training_set_dilated_ring_nc = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_nc_dilated_ring.csv')
        feature_file_training_set_dilated_ring_delay = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_delay_dilated_ring.csv')

        feature_file_training_set_eroded_ring_pvp = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_pvp_eroded_ring.csv')
        feature_file_training_set_eroded_ring_art = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_art_eroded_ring.csv')
        feature_file_training_set_eroded_ring_nc = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_nc_eroded_ring.csv')
        feature_file_training_set_eroded_ring_delay = os.path.join(file_root_path, f'fold_{fold_index}_trainval_features_delay_eroded_ring.csv')
        n_header_cols = 38


        if do_extension:
            df_features_training = radiomics_data_parse(feature_file_training_set_pvp, feature_file_training_set_art, 
                                                        feature_file_training_set_nc, feature_file_training_set_delay,
                                                        feature_file_training_set_dilated_ring_pvp, feature_file_training_set_dilated_ring_art,
                                                        feature_file_training_set_dilated_ring_nc, feature_file_training_set_dilated_ring_delay
                                                        )

        else:
            if phases_num == 4:
                df_features_training = radiomics_data_parse(feature_file_training_set_pvp, feature_file_training_set_art, 
                                                            feature_file_training_set_nc, feature_file_training_set_delay)
            elif phases_num == 3:
                df_features_training = radiomics_data_parse(feature_file_training_set_pvp, feature_file_training_set_art, 
                                                            feature_file_training_set_nc)


        label_file_training_set = os.path.join(file_root_path, f'fold_{fold_index}_trainval_tumor_label.txt')
        labels_training = np.loadtxt(label_file_training_set)

        # Identify and remove strongly correlated features in the training dataset
        df_features_training_uncorr, dropped_features = remove_correlated_features(df_features_training, corr_threshold=corr_threshold)
        dropped_features = np.asarray(dropped_features)

        if do_extension:
            os.makedirs(f'feature_selection/{exp}/{fold_name}/fold_{fold_index}',exist_ok=True)
            np.savetxt(f'feature_selection/{exp}/{fold_name}/fold_{fold_index}/dropped_features.csv', dropped_features, newline=',', fmt='%s')
        else:
            os.makedirs(f'feature_selection/{exp}/{fold_name}_without_extension/fold_{fold_index}',exist_ok=True)
            np.savetxt(f'feature_selection/{exp}/{fold_name}_without_extension/fold_{fold_index}/dropped_features.csv', dropped_features, newline=',', fmt='%s')

        feature_names = df_features_training_uncorr.columns.to_numpy()

        X_train = df_features_training_uncorr.to_numpy()
        y_train = labels_training

        y_train = np.array([int(item) for item in labels_training])

        rf = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=0)
        rf.fit(X_train, y_train)

        ### internal validation
        y_pred_oob = np.argmax(rf.oob_decision_function_, axis=1)
        oob_accuracy = accuracy_score(y_train, y_pred_oob)
        oob_report = classification_report(y_train, y_pred_oob)

        print("Internal out-of-bag (OOB) validation:")
        print('oob_accuracy: {:.2f}'.format(oob_accuracy))
        print("\nClassification Report:\n", oob_report)

        importances, _ = extract_feature_importance(rf)
        ranked_features_id, ranked_features_name, ranked_importances = rank_feature_importance(importances, feature_names, n=10)
        plot_feature_importance(ranked_features_id, ranked_importances)

