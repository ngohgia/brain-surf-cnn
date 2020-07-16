#!/bin/bash

cd /home/ghn8/TaskContrastFromRest/clean_code

source activate gia_graphite_cuda10;

for SEED in $(seq 3 20);
do
  echo $SEED
  OUT_DIR=/nfs03/users/ghn8/TaskContrastsFromRest/random_parcellation/seed${SEED}_feat64_s8_c48_lr0.01_seed28
  python -u compute_within_and_across_subj_loss.py \
         --subj_list /home/ghn8/TaskContrastFromRest/clean_code/data/MICCAI2020/sample_HCP_train_val_subj_ids.csv \
         --pred_dir $OUT_DIR/predict_on_train_subj/train \
         --contrast_dir /nfs03/data/HCP_200113/derived/preprocessed/contrasts \
         --mask /home/ghn8/TaskContrastFromRest/clean_code/data/glasser_medial_wall_mask.npy
done
