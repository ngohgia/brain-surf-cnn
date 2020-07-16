import numpy as np

num_samples = 100
train_subj_ids = np.genfromtxt('/home/ghn8/TaskContrastFromRest/clean_code/data/MICCAI2020/HCP_train_val_subj_ids.csv', dtype="<U13")
sample_subj_ids = np.random.choice(train_subj_ids, size=num_samples, replace=False)

np.savetxt('/home/ghn8/TaskContrastFromRest/clean_code/data/MICCAI2020/sample_HCP_train_val_subj_ids.csv', sample_subj_ids, fmt="%s")
