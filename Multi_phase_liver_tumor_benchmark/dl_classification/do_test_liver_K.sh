#!/bin/sh


export PYTHONPATH='/REPLACEMENT_PATH/models':/REPLACEMENT_PATH/public-project/REPLACEMENT_USER_PROJECT
exp=$(python -c "import my_globals; print(my_globals.exp)")
fold_index=$(python -c "import my_globals; print(my_globals.fold_index)")
fold_name=$(python -c "import my_globals; print(my_globals.fold_name)")
date=$(python -c "import my_globals; print(my_globals.date)")


# # ##cnn2d###############################################################################################################################################
for fold in {1..5}
do
  CUDA_VISIBLE_DEVICES=1 python validate.py \
    --data_dir ../tumor_radiomics/Label/crop_tumor/total/image \
    --val_anno_file ../tumor_radiomics/Label/exp/$exp/$fold_name/fold_${fold}_test_tumor_label_dl.txt \
    --b 2 --model cnn2d \
    --num-classes 4 \
    --img_size 20 160 160 \
    --crop_size 20 160 160 \
    --checkpoint ./tumor_DL/dl_cls_3d/main/output/0307/multi_cls_4_dataset_essay/fold_${fold}/train/${fill_the_blank}-cnn2d/model_best.pth.tar \
    --results-dir ./output/0307/$exp/fold_${fold}/test/cnn2d &
done
wait
# # ###############################################################################################################################################

# ##CNN3D###############################################################################################################################################
for fold in {1..5}
do
  python validate.py \
    --data_dir ../tumor_radiomics/Label/crop_tumor/total/image \
    --val_anno_file ../tumor_radiomics/Label/exp/$exp/$fold_name/fold_${fold}_test_tumor_label_dl.txt \
    --b 2 --model resnet34 \
    --num-classes 4 \
    --img_size 20 160 160 \
    --crop_size 20 160 160 \
    --checkpoint ./tumor_DL/dl_cls_3d/main/output/0307/multi_cls_4_dataset_essay/fold_${fold}/train/${fill_the_blank}-resnet34/model_best.pth.tar \
    --results-dir ./output/0307/$exp/fold_${fold}/test/Med3D &
done
wait
##########################################################################################################################################################



# ##Uniformer(Phase-variant)###############################################################################################################################################
for fold in {1..5}
do
  CUDA_VISIBLE_DEVICES=1 python validate.py \
    --data_dir ../tumor_radiomics/Label/crop_tumor/total/image \
    --val_anno_file ../tumor_radiomics/Label/exp/$exp/$fold_name/fold_${fold}_test_tumor_label_dl.txt \
    --b 2 --model uniformer_base_IL \
    --num-classes 4 \
    --img_size 20 160 160 \
    --crop_size 20 160 160 \
    --checkpoint ./tumor_DL/dl_cls_3d/main/output/0307/multi_cls_4_dataset_essay/fold_${fold}/train/${fill_the_blank}-uniformer_base_IL/model_best.pth.tar \
    --results-dir ./output/0307/$exp/fold_${fold}/test/uniformer &
done
wait
# ###############################################################################################################################################





