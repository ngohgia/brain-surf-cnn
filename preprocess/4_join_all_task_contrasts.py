import os
import numpy as np
from argparse import ArgumentParser

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import CONTRASTS

parser = ArgumentParser()

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
        print(i+1, "/", len(subj_ids))
        subj = subj_ids[i]
    
        subj_task_data = []
        subj_task_data_file = os.path.join(contrasts_dir, "%s_joint_LR_all_tasks.npy" % subj)
        for item in CONTRASTS:
            task, cope, c = item
        
            lh_task_file = os.path.join(args.input_dir, task, subj, "cope%d.feat" % cope, "zstat1.L.func.gii")
            rh_task_file = os.path.join(args.input_dir, task, subj, "cope%d.feat" % cope, "zstat1.R.func.gii")
        
            if os.path.exists(lh_task_file) and os.path.exists(rh_task_file):
                lh_data = np.expand_dims(nib.gifti.read(lh_task_file).darrays[0].data, axis=0)
                rh_data = np.expand_dims(nib.gifti.read(rh_task_file).darrays[0].data, axis=0)
    
                joint_LR_data = np.concatenate((lh_data, rh_data))
                subj_task_data.append(joint_LR_data)
    
        np.save(subj_task_data_file, subj_task_data)
