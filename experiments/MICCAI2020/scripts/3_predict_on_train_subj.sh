#!/bin/bash

source activate brain_surf_cnn

PROJECT_DIR=/home/ghn8/brain_surf_cnn
cd $PROJECT_DIR

MICCAI_DIR=$PROJECT_DIR/experiments/MICCAI2020
DATA_DIR=$MICCAI_DIR/sample_data
NUM_ICS=50
NUM_SAMPLES=8

RSFC_DIR=$DATA_DIR/rsfc_d${NUM_ICS}_sample$NUM_SAMPLES

SUBJ_LIST_FILE=$MICCAI_DIR/sample_HCP_train_val_subj_ids.csv

MESH_TEMPLATES_DIR=$PROJECT_DIR/data/fs_LR_mesh_templates

OUTPUTS_DIR=$MICCAI_DIR/sample_outputs/HCP_sample_feat64_s${NUM_SAMPLES}_c${NUM_ICS}_lr0.01_seed28

python -u predict.py \
       --gpus 0 \
       --ver best_corr \
       --subj_list $SUBJ_LIST_FILE \
       --rsfc_dir $RSFC_DIR \
       --mesh_dir $MESH_TEMPLATES_DIR \
       --checkpoint_file $OUTPUTS_DIR/best_corr.pth \
       --save_dir $OUTPUTS_DIR/predict_on_train_subj
