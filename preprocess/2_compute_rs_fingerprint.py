import nibabel as nib
from nibabel import freesurfer as fs
import os
import numpy as np
import matplotlib.pyplot as plt
from nibabel import cifti2
import nibabel.gifti as gi
import pickle

import unittest
import torch
import torch.nn as nn
from argparse import ArgumentParser

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import compute_corr_coeff

parser = ArgumentParser()

parser.add_argument("--input_dir",
                    type=str,
                    help="Directory containing the HCP subjects' resting-state fMRI timeseries")

parser.add_argument("--subj_ids",
                    type=str,
                    help="File containing HCP subject IDs, one subject per line")

parser.add_argument("--node_ts_dir",
                    type=str,
                    help="Directory containing subject-specific HCP ICA node timeseries")

parser.add_argument("--output_dir",
                    type=str,
                    help="Outuput director for the rsfc fingerprints")

parser.add_argument("--num_ics",
                    type=int,
                    default=50,
                    help="Number of HCP ICA components to use")

parser.add_argument("--num_samples",
                    type=int,
                    default=8,
                    help="Number of fingerprint samples generated per subject")

def extract_ts_from_mesh(mesh_file, num_ts=1200):
    mesh = gi.read(mesh_file)
    data = []

    if (num_ts != len(mesh.darrays)):
        print(len(mesh.darrays))
    assert(num_ts == len(mesh.darrays))
    for i in range(num_ts):
        data.append(mesh.darrays[i].data)
    data = np.asarray(data).T
    return data

if __name__ == "__main__":
    args = parser.parse_args()

    HCP_RS_LENGTH = 4800
    sample_length = HCP_RS_LENGTH // args.num_samples
    
    subj_ids = np.genfromtxt(args.subj_ids, dtype='<U20')
    rsfc_dir = os.path.join(args.output_dir, "rsfc_d%d_sample%d" % (args.num_ics, args.num_samples))
    os.makedirs(rsfc_dir, exist_ok=True)

    subj_ids = ["352132"]
    
    for j in range(len(subj_ids)):
        print(j+1, "/", len(subj_ids))
    
        subj_id = subj_ids[j]
        
        lh_subj_rest1_lr_file = os.path.join(args.input_dir, 'rfMRI_REST1_LR', subj_id, 'rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.L.func.gii')
        lh_subj_rest1_rl_file = os.path.join(args.input_dir, 'rfMRI_REST1_RL', subj_id, 'rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.L.func.gii')
        lh_subj_rest2_lr_file = os.path.join(args.input_dir, 'rfMRI_REST2_LR', subj_id, 'rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.L.func.gii')
        lh_subj_rest2_rl_file = os.path.join(args.input_dir, 'rfMRI_REST2_RL', subj_id, 'rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.L.func.gii')
    
        rh_subj_rest1_lr_file = os.path.join(args.input_dir, 'rfMRI_REST1_LR', subj_id, 'rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.R.func.gii')
        rh_subj_rest1_rl_file = os.path.join(args.input_dir, 'rfMRI_REST1_RL', subj_id, 'rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.R.func.gii')
        rh_subj_rest2_lr_file = os.path.join(args.input_dir, 'rfMRI_REST2_LR', subj_id, 'rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.R.func.gii')
        rh_subj_rest2_rl_file = os.path.join(args.input_dir, 'rfMRI_REST2_RL', subj_id, 'rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.R.func.gii')
    
        subj_node_ts_file = os.path.join(args.node_ts_dir, "%s.txt" % subj_id)
                
        if os.path.exists(subj_node_ts_file) and os.path.exists(lh_subj_rest1_lr_file) and os.path.exists(lh_subj_rest1_rl_file) and os.path.exists(lh_subj_rest2_lr_file) and os.path.exists(lh_subj_rest2_rl_file):
            # try:
            lh_subj_rest1_lr_data = extract_ts_from_mesh(lh_subj_rest1_lr_file)
            lh_subj_rest1_rl_data = extract_ts_from_mesh(lh_subj_rest1_rl_file)
            lh_subj_rest2_lr_data = extract_ts_from_mesh(lh_subj_rest2_lr_file)
            lh_subj_rest2_rl_data = extract_ts_from_mesh(lh_subj_rest2_rl_file)
    
            rh_subj_rest1_lr_data = extract_ts_from_mesh(rh_subj_rest1_lr_file)
            rh_subj_rest1_rl_data = extract_ts_from_mesh(rh_subj_rest1_rl_file)
            rh_subj_rest2_lr_data = extract_ts_from_mesh(rh_subj_rest2_lr_file)
            rh_subj_rest2_rl_data = extract_ts_from_mesh(rh_subj_rest2_rl_file)
    
            lh_data = np.concatenate((lh_subj_rest1_lr_data, lh_subj_rest1_rl_data, lh_subj_rest2_lr_data, lh_subj_rest2_rl_data), axis=1)
            rh_data = np.concatenate((rh_subj_rest1_lr_data, rh_subj_rest1_rl_data, rh_subj_rest2_lr_data, rh_subj_rest2_rl_data), axis=1)
            
            subj_node_ts = np.genfromtxt(subj_node_ts_file).T
    
            # for i in range(arsg.num_samples):
            for i in range(4, 5):
                subj_rsfc_file = os.path.join(args.output_dir, "joint_LR_%s_sample%d_rsfc.npy" % (subj_id, i))
   
                print(subj_rsfc_file) 
                if not os.path.exists(subj_rsfc_file):
                    ts = np.arange(i*sample_length, (i+1)*sample_length)
                    lh_subj_conn = compute_corr_coeff(lh_data[:, ts], subj_node_ts[:, ts])
                    rh_subj_conn = compute_corr_coeff(rh_data[:, ts], subj_node_ts[:, ts])
                    lh_subj_conn[np.isnan(lh_subj_conn)] = 0
                    rh_subj_conn[np.isnan(rh_subj_conn)] = 0
    
                    subj_rsfc_conn = np.concatenate((lh_subj_conn.T, rh_subj_conn.T))
                    np.save(subj_rsfc_file, subj_rsfc_conn)
    
            # except:
            #    print("Error", subj_id)
