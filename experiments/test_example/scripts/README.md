## Scripts for the test experiment

In all the scripts, please replace `PROJECT_DIR` to point to the root directory of this repository. `submit.sh` is also provided for Slurm job queue.
Please run the scripts in the following order:
- `0_run_preprocess_pipeline.sh` preprocesses the downloaded HCP data to be compatible with BrainSurfCNN. The following variables need to be updated according to your local environment:
  - `WB_COMMAND` points to the HCP workbench binary `wb_command`
  - `HCP_DATA_DIR` points to the downloaded HCP data.
  - `HCP_NODE_TS_DIR` points to a HCP node timeseries, e.g. `3T_HCP1200_MSMAll_d50_ts2`
 
- `1_train.sh` trains the model with MSE Loss.
- `2_predict.sh` uses the saved checkpoint (trained with MSE) to predict on test subjects.
- `3_predict_on_train_subj.sh` makes prediction on training subjects to compute within and across subject errors, which are hyperparameters needed for finetuning the model using reconstructive-contrastive (RC) loss.
- `4_compute_within_and_across_subj_loss.sh` computes the within and across subject errors.
- `5_finetune_with_rc_loss.sh` finetunes the model with RC loss.
- `2_predict.sh` uses the finetuned checkpoint to make prediction. You would need to change the path to the checkpoint.
