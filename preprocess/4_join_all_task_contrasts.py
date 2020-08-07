import os
import numpy as np
import nibabel.gifti as gi
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import sys
sys.path.insert(0, "..")
from utils.utilities import CONTRASTS

parser = ArgumentParser(description="Compute resting-state fingerprints from timeseries in cifti format",
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--input_dir",
                    type=str,
                    help="Directory containing outputs from 3_separate_task_cifti.sh")

parser.add_argument("--subj_ids",
                    type=str,
                    help="File containing HCP subject IDs, one subject per line")

parser.add_argument("--output_dir",
                    type=str,
                    help="Outuput director for the joint task contrasts")

if __name__ == "__main__":
    args = parser.parse_args()

    subj_ids = np.genfromtxt(args.subj_ids, dtype='<U20')
    contrasts_dir = os.path.join(args.output_dir, "contrasts")
    os.makedirs(contrasts_dir, exist_ok=True)

    for i in range(len(subj_ids)):
        subj = subj_ids[i]
        print(i+1, "/", len(subj_ids), ":", subj)
    
        subj_task_data = []
        subj_task_data_file = os.path.join(contrasts_dir, "%s_joint_LR_task_contrasts.npy" % subj)
        for item in CONTRASTS:
            task, cope, c = item
        
            lh_task_file = os.path.join(args.input_dir, subj, task, "cope%d.feat" % cope, "zstat1.L.func.gii")
            rh_task_file = os.path.join(args.input_dir, subj, task, "cope%d.feat" % cope, "zstat1.R.func.gii")
       
            if os.path.exists(lh_task_file) and os.path.exists(rh_task_file):
                lh_data = gi.read(lh_task_file).darrays[0].data
                rh_data = gi.read(rh_task_file).darrays[0].data
    
                # joint_LR_data = np.concatenate((lh_data, rh_data), axis=-1)
                subj_task_data.append(lh_data)
                subj_task_data.append(rh_data)
    
        np.save(subj_task_data_file, subj_task_data)
