#!/bin/bash

source activate brain_surf_cnn

PROJECT_DIR=/home/ghn8/brain_surf_cnn
cd $PROJECT_DIR

WORK_DIR=$PROJECT_DIR/experiments/test_example
DATA_DIR=$WORK_DIR/sample_data
NUM_ICS=50
NUM_SAMPLES=8

RSFC_DIR=$DATA_DIR/rsfc_d${NUM_ICS}_sample$NUM_SAMPLES

SUBJ_LIST_FILE=$WORK_DIR/sample_HCP_test_retest_subj_ids.csv

MESH_TEMPLATES_DIR=$PROJECT_DIR/data/fs_LR_mesh_templates

# OUTPUTS_DIR=$WORK_DIR/sample_outputs/test_example_feat64_s${NUM_SAMPLES}_c${NUM_ICS}_lr0.01_seed28
OUTPUTS_DIR=$WORK_DIR/sample_outputs/test_example_feat64_s${NUM_SAMPLES}_c${NUM_ICS}_lr0.01_seed28/finetuned_feat64_s${NUM_SAMPLES}_c${NUM_ICS}_lr0.01_seed28 # finetuned

python -u predict.py \
       --gpus 0 \
       --ver best_corr \
       --subj_list $SUBJ_LIST_FILE \
       --rsfc_dir $RSFC_DIR \
       --mesh_dir $MESH_TEMPLATES_DIR \
       --checkpoint_file $OUTPUTS_DIR/best_corr.pth \
       --save_dir $OUTPUTS_DIR/predict
