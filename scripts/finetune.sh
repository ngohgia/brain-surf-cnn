#!/bin/bash

cd /home/ghn8/TaskContrastFromRest/clean_code

source activate gia_graphite_cuda10;

SEED=$1

# --rsfc_dir /share/sablab/nfs03/data/HCP_200113/derived/rsFC_d50_sample8 \
# --rsfc_dir /share/sablab/nfs03/data/HCP_200113/derived/code/from_ICA_parcel/rsfc_random_d50_seed$SEED \
python -u train.py \
       --gpus 0 \
       --ver finetuned \
       --loss rc \
       --checkpoint_file /share/sablab/nfs03/users/ghn8/TaskContrastsFromRest/debug/ch50_scheduled_feat64_s8_c50_lr0.01_seed28/best_corr.pth \
       --subj_list /home/ghn8/TaskContrastFromRest/clean_code/data/MICCAI2020/HCP_train_val_subj_ids.csv \
       --rsfc_dir /share/sablab/nfs03/data/HCP_200113/derived/preprocessed/rsfc_d50_sample8/ \
       --contrast_dir /share/sablab/nfs03/data/HCP_200113/derived/preprocessed/contrasts \
       --mesh_dir /home/ghn8/TaskContrastFromRest/mesh_models/mesh_files \
       --save_dir /share/sablab/nfs03/users/ghn8/TaskContrastsFromRest/debug/ch50_scheduled_feat64_s8_c50_lr0.01_seed28 \
       --n_val_subj 50 \
       --n_channels_per_hemi 50
