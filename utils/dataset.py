import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

class MultipleSampleMeshDataset(Dataset):
    def __init__(self, subj_ids, contrasts, rsfc_dir, contrast_dir, num_samples=8):
        self.rsfc_dir = rsfc_dir
        self.contrast_dir = contrast_dir
        self.subj_ids = subj_ids
        self.contrasts = contrasts
        self.num_samples = num_samples

    def __getitem__(self, index):
        subj = self.subj_ids[index]

        sample_id = np.random.randint(0, self.num_samples)
        rsfc_file = os.path.join(self.rsfc_dir, "joint_LR_%s_sample%d_rsfc.npy" % (subj, sample_id))
        subj_rsfc_data = np.load(rsfc_file).T

        subj_task_data = np.load(os.path.join(self.contrast_dir, "%s_joint_LR_task_contrasts.npy" % subj))
        # subj_task_data = np.load(os.path.join(self.contrast_dir, "%s_joint_LR_all_tasks.npy" % subj))
        # subj_task_data = subj_task_data[[25, 51], :]

        return torch.cuda.FloatTensor(subj_rsfc_data) , torch.cuda.FloatTensor(subj_task_data)

    def __len__(self):
        return len(self.subj_ids)
