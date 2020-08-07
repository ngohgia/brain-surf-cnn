#!/bin/bash

WB_COMMAND=/share/sablab/nfs03/users/ghn8/lib/workbench/bin_linux64/wb_command

HCP_DATA_DIR=/share/sablab/nfs03/data/HCP_200113

# train-validation subjects
HCP_SUBJ_IDS=${PWD}/sample_HCP_train_val_subj_ids.csv # Replace by ../../data/MICCAI2020/HCP_train_val_subj_ids.csv to replicate MICCAI2020 results

# test-retest subjects
# HCP_SUBJ_IDS=${PWD}/sample_HCP_test_retest_subj_ids.csv # Replace by ../../data/MICCAI2020/HCP_test_retest_subj_ids.csv to replicate MICCAI2020 results

HCP_NODE_TS_DIR=$HCP_DATA_DIR/HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_d50_ts2

DATA_OUTPUT_DIR=${PWD}/sample_data
GIFTI_DIR=$DATA_OUTPUT_DIR/gifti

NUM_ICS=50
NUM_SAMPLES=8

PREPROCESS_DIR=../../../preprocess

cd $PREPROCESS_DIR

echo "Step 1: Separating resting-state timeseries ciftis"
bash 1_separate_rs_cifti.sh $WB_COMMAND $HCP_DATA_DIR $HCP_SUBJ_IDS $GIFTI_DIR

echo "Step 2: Compute resting-state fingerprints"
python -u 2_compute_rs_fingerprint.py \
    --input_dir $GIFTI_DIR \
    --subj_ids $HCP_SUBJ_IDS \
    --node_ts_dir $HCP_NODE_TS_DIR \
    --num_ics $NUM_ICS \
    --num_samples $NUM_SAMPLES \
    --output_dir $DATA_OUTPUT_DIR

echo "Step 3: Seperating task ciftis"
bash 3_separate_task_cifti.sh $WB_COMMAND $HCP_DATA_DIR $HCP_SUBJ_IDS $GIFTI_DIR

echo "Step 4: Join each subject's task contrasts into a single multi-channel volume"
python -u 4_join_all_task_contrasts.py \
    --input_dir $GIFTI_DIR \
    --subj_ids $HCP_SUBJ_IDS \
    --output_dir $DATA_OUTPUT_DIR
