# Path to binary for wb_command
WB_COMMAND=/share/sablab/nfs03/users/ghn8/lib/workbench/bin_linux64/wb_command

# Path to director containing HCP cifti timeseries
HCP_DIR=/share/sablab/nfs03/data/HCP_200113
OUTPUT_DIR=gifti

# Path to HCP subject IDs
SUBJLIST=sample_subj_ids.txt

while read -r SUBJECT;
do
    echo $SUBJECT
    for SESSION in rfMRI_REST1_LR rfMRI_REST1_RL rfMRI_REST2_LR rfMRI_REST1_RL;
    do
        SUBJ_DIR=$HCP_DIR/$SUBJECT/
        SUBJ_OUT_DIR=$OUTPUT_DIR/$SUBJECT/
        
        mkdir -p $SUBJ_OUT_DIR

        CIFTI_FILE=$SUBJ_DIR/MNINonLinear/Results/$SESSION/${SESSION}_Atlas_MSMAll_hp2000_clean.dtseries.nii

        LH_NIFTI_FILE=$SUBJ_OUT_DIR/${SESSION}_Atlas_MSMAll_hp2000_clean.L.func.gii
        RH_NIFTI_FILE=$SUBJ_OUT_DIR/${SESSION}_Atlas_MSMAll_hp2000_clean.R.func.gii
        SUBCORTICAL_FILE=$SUBJ_OUT_DIR/${SESSION}_Atlas_MSMAll_hp2000_clean.subcortical.nii.gz

        if [ -f "$CIFTI_FILE" ] && [ ! -f "$LH_NIFTI_FILE" ]; then
            echo "  $SESSION"
            $WB_COMMAND -cifti-separate  $CIFTI_FILE COLUMN -metric CORTEX_LEFT $LH_NIFTI_FILE -metric CORTEX_RIGHT $RH_NIFTI_FILE -volume-all $SUBCORTICAL_FILE
        fi
    done
done < $SUBJLIST
