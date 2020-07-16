# Path to binary for wb_command
WB_COMMAND=/share/sablab/nfs03/users/ghn8/lib/workbench/bin_linux64/wb_command

# Path to director containing HCP S1200 subject data
HCP_DIR=/share/sablab/nfs03/data/HCP_200113
OUTPUT_DIR=gifti

# Path to HCP subject IDs
SUBJLIST=sample_subj_ids.txt



declare -A TASK_COPEIDS=( ["LANGUAGE"]="1 2 3" ["RELATIONAL"]="1 2 3" ["SOCIAL"]="1 2 6" ["EMOTION"]="1 2 3" \
                          ["WM"]="1 2 3 4 5 6 7 8 9 10 11 15 16 17 18 19 20 21 22" \
                          ["MOTOR"]="1 2 3 4 5 6 7 8 9 10 11 12 13" \
                          ["GAMBLING"]="1 2 3" )

while read -r SUBJECT;
do
    echo $SUBJECT
    SUBJ_DIR=$HCP_DIR/$SUBJECT/

    for TASK in "${!TASK_COPEIDS[@]}";
    do
        declare -a COPEIDS="${TASK_COPEIDS[$TASK]}"
        for COPEID in $COPEIDS
        do
            SUBJ_OUT_DIR=$OUTPUT_DIR/$SUBJECT/$TASK/cope$COPEID.feat
            mkdir -p $SUBJ_OUT_DIR

            CIFTI_FILE=$SUBJ_DIR/MNINonLinear/Results/tfMRI_$TASK/tfMRI_${TASK}_hp200_s2_level2_MSMAll.feat/GrayordinatesStats/cope${COPEID}.feat/zstat1.dtseries.nii

            LH_NIFTI_FILE=$SUBJ_OUT_DIR/zstat1.L.func.gii
            RH_NIFTI_FILE=$SUBJ_OUT_DIR/zstat1.R.func.gii
            SUBCORTICAL_FILE=$SUBJ_OUT_DIR/zstat1.subcortical.nii.gz

            echo $CIFTI_FILE
            if [ -f "$CIFTI_FILE" ] && [ ! -f "$LH_NIFTI_FILE" ]; then
                echo "$WB_COMMAND -cifti-separate  $CIFTI_FILE COLUMN -metric CORTEX_LEFT $LH_NIFTI_FILE -metric CORTEX_RIGHT $RH_NIFTI_FILE -volume-all $SUBCORTICAL_FILE"
                $WB_COMMAND -cifti-separate  $CIFTI_FILE COLUMN -metric CORTEX_LEFT $LH_NIFTI_FILE -metric CORTEX_RIGHT $RH_NIFTI_FILE -volume-all $SUBCORTICAL_FILE
            fi
        done
    done
done < $SUBJLIST
