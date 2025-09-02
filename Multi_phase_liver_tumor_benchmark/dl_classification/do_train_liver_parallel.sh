#!/bin/sh

trap "kill 0" SIGINT

export PYTHONPATH='/REPLACEMENT_PATH/models':/REPLACEMENT_PATH/public-project/REPLACEMENT_USER_PROJECT
exp=$(python -c "import my_globals; print(my_globals.exp)")
fold_index=$(python -c "import my_globals; print(my_globals.fold_index)")
fold_name=$(python -c "import my_globals; print(my_globals.fold_name)")


for model in cnn2d resnet34 uniformer_base_IL; do
    for fold in {1..5}; do
        case $model in
            resnet34)
                pretrained_weights="./weights/resnet_34_23dataset.pth"
                ;;
            uniformer_base_IL)
                pretrained_weights="./weights/uniformer_base_k400_8x8_partial.pth"
                ;;
        esac

        CUDA_VISIBLE_DEVICES=$((fold-1)) python train.py \
        --data_dir ../tumor_radiomics/Label/crop_tumor/total/image \
        --feat_csv_dir ../tumor_radiomics/Label/exp/$exp/${fold_name}/fold_${fold}_scaled_features_dl.csv \
        --train_anno_file ../tumor_radiomics/Label/exp/$exp/${fold_name}/fold_${fold}_train_tumor_label_dl.txt \
        --val_anno_file ../tumor_radiomics/Label/exp/$exp/${fold_name}/fold_${fold}_val_tumor_label_dl.txt \
        --batch-size 16 --model $model \
        --lr 1e-4 --warmup-epochs 20 --epochs 70 --num-classes 4 \
        --img_size 20 160 160 \
        --crop_size 20 160 160 \
        --output ./output/0307/$exp/fold_${fold}/train &
    done
    wait
done
