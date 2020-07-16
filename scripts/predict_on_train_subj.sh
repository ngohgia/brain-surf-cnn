#!/bin/bash

cd /home/ghn8/TaskContrastFromRest/clean_code

source activate gia_graphite_cuda10;

OUT_DIR=/share/sablab/nfs03/users/ghn8/TaskContrastsFromRest/debug/ch50_feat64_s8_c50_lr0.01_seed28

python -u predict.py \
       --gpus 0 \
       --ver best_corr \
       --subj_list /home/ghn8/TaskContrastFromRest/clean_code/data/MICCAI2020/HCP_train_val_subj_ids.csv \
       --rsfc_dir /share/sablab/nfs03/data/HCP_200113/derived/preprocessed/rsfc_d50_sample8 \
       --mesh_dir /home/ghn8/TaskContrastFromRest/mesh_models/mesh_files \
       --checkpoint_file $OUT_DIR/best_corr.pth \
       --save_dir $OUT_DIR/predict_on_train_subj
