# bash 1_separate_rs_cifti.sh

python -u 2_compute_rs_fingerprint.py \
    --input_dir /share/sablab/nfs03/data/HCP_200113/derived \
    --subj_ids sample_subj_ids.txt \
    --node_ts_dir /share/sablab/nfs03/data/HCP_200113/HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_d50_ts2 \
    --output_dir /share/sablab/nfs03/data/HCP_200113/derived/preprocessed/rsfc_d50_sample8
    # --output_dir outputs

# bash 3_separate_task_cifti.sh
# 
# python -u 4_join_all_task_contrasts.py \
#     --input_dir gifti \
#     --subj_ids sample_subj_ids.txt \
#     --output_dir outputs
