#!/bin/bash

source activate brain_surf_cnn

PROJECT_DIR=/home/ghn8/brain_surf_cnn
cd $PROJECT_DIR/utils

MICCAI_DIR=$PROJECT_DIR/experiments/MICCAI2020
DATA_DIR=$MICCAI_DIR/sample_data
NUM_ICS=50
NUM_SAMPLES=8

CONTRASTS_DIR=$DATA_DIR/contrasts

SUBJ_LIST_FILE=$MICCAI_DIR/sample_HCP_train_val_subj_ids.csv

OUTPUTS_DIR=$MICCAI_DIR/sample_outputs/HCP_sample_feat64_s${NUM_SAMPLES}_c${NUM_ICS}_lr0.01_seed28

python -u compute_within_and_across_subj_loss.py \
       --subj_list $SUBJ_LIST_FILE \
       --pred_dir $OUTPUTS_DIR/predict_on_train_subj/best_corr \
       --contrast_dir $CONTRASTS_DIR \
       --mask $PROJECT_DIR/data/glasser_medial_wall_mask.npy
