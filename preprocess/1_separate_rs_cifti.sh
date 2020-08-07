#!/bin/bash

# Path to binary for wb_command
WB_COMMAND=$1

# Path to directory containing HCP cifti timeseries
HCP_DIR=$2

# Path to HCP subject IDs
SUBJLIST=$3

OUTPUT_DIR=$4

while read -r SUBJECT;
do
    echo $SUBJECT
    for SESSION in rfMRI_REST1_LR rfMRI_REST1_RL rfMRI_REST2_LR rfMRI_REST2_RL;
    do
        SUBJ_DIR=$HCP_DIR/$SUBJECT/
        SUBJ_OUT_DIR=$OUTPUT_DIR/$SUBJECT/
        
        mkdir -p $SUBJ_OUT_DIR

        CIFTI_FILE=$SUBJ_DIR/MNINonLinear/Results/$SESSION/${SESSION}_Atlas_MSMAll_hp2000_clean.dtseries.nii

        LH_CIFTI_FILE=$SUBJ_OUT_DIR/${SESSION}_Atlas_MSMAll_hp2000_clean.L.func.gii
        RH_CIFTI_FILE=$SUBJ_OUT_DIR/${SESSION}_Atlas_MSMAll_hp2000_clean.R.func.gii
        SUBCORTICAL_FILE=$SUBJ_OUT_DIR/${SESSION}_Atlas_MSMAll_hp2000_clean.subcortical.nii.gz

        if [ -f "$CIFTI_FILE" ] && [ ! -f "$LH_CIFTI_FILE" ]; then
            echo "  $SESSION"
            $WB_COMMAND -cifti-separate  $CIFTI_FILE COLUMN -metric CORTEX_LEFT $LH_CIFTI_FILE -metric CORTEX_RIGHT $RH_CIFTI_FILE -volume-all $SUBCORTICAL_FILE
        fi
    done
done < $SUBJLIST
