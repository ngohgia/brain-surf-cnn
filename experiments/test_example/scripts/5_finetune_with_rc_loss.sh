#!/bin/bash

source activate brain_surf_cnn

PROJECT_DIR=/home/ghn8/brain_surf_cnn
cd $PROJECT_DIR

NUM_ICS=50
NUM_SAMPLES=8
NUM_VAL_SUBJ=5

WORK_DIR=$PROJECT_DIR/experiments/test_example
DATA_DIR=$WORK_DIR/sample_data
RSFC_DIR=$DATA_DIR/rsfc_d${NUM_ICS}_sample$NUM_SAMPLES
CONTRASTS_DIR=$DATA_DIR/contrasts

SUBJ_LIST_FILE=$WORK_DIR/sample_HCP_train_val_subj_ids.csv

MESH_TEMPLATES_DIR=$PROJECT_DIR/data/fs_LR_mesh_templates

OUTPUTS_DIR=$WORK_DIR/sample_outputs

python -u train.py \
       --gpus 0 \
       --ver finetuned \
       --loss rc \
       --checkpoint_file $OUTPUTS_DIR/test_example_feat64_s${NUM_SAMPLES}_c${NUM_ICS}_lr0.01_seed28/best_corr.pth \
       --subj_list $SUBJ_LIST_FILE \
       --rsfc_dir $RSFC_DIR \
       --contrast_dir $CONTRASTS_DIR \
       --mesh_dir $MESH_TEMPLATES_DIR \
       --save_dir $OUTPUTS_DIR/test_example_feat64_s${NUM_SAMPLES}_c${NUM_ICS}_lr0.01_seed28 \
       --n_val_subj $NUM_VAL_SUBJ \
       --init_within_subj_margin 6 \
       --init_across_subj_margin 7 \
