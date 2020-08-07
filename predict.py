import os
import numpy as np

import torch
from torch.nn.parameter import Parameter

from model.brain_surf_cnn import BrainSurfCNN
from utils.parser import test_args
from utils.dataset import MultipleSampleMeshDataset
from utils.utilities import CONTRASTS, save_checkpoint


if __name__ == "__main__":

    args = test_args()

    """Init"""
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus

    output_dir = os.path.join(args.save_dir, args.ver)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise Exception("Output dir exists: %s" % output_dir)


    """Load Data"""
    subj_ids = np.genfromtxt(args.subj_list, dtype="<U13")

    model = BrainSurfCNN(
        mesh_dir=args.mesh_dir,
        in_ch=args.n_channels_per_hemi*2,
        out_ch=args.n_output_channels*2,
        max_level=args.max_mesh_lvl,
        min_level=args.min_mesh_lvl,
        fdim=args.n_feat_channels)
    model.cuda()

    state_dict = torch.load(args.checkpoint_file)['state_dict']
    model_state_dict = model.state_dict()
    
    for name, param in state_dict.items():
        if name not in model_state_dict:
            continue
        if 'none' in name:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        model_state_dict[name].copy_(param)
    
    model.eval()

    with torch.no_grad():
        for i in range(len(subj_ids)):
            print(i+1, "/", len(subj_ids))
            subj = subj_ids[i]
            pred_file = os.path.join(output_dir, "%s_pred.npy" % subj)
            if not os.path.exists(pred_file):
                subj_pred = []
                for sample_id in range(args.n_samples_per_subj):
                    rsfc_file = os.path.join(args.rsfc_dir, "joint_LR_%s_sample%d_rsfc.npy" % (subj, sample_id))
                    subj_rsfc_data = torch.cuda.FloatTensor(np.expand_dims(np.load(rsfc_file), axis=0))

                    sample_pred = model(subj_rsfc_data)
                    
                    subj_pred.append(sample_pred.cpu().detach().numpy().squeeze(0))
                subj_pred = np.asarray(subj_pred)
                np.save(pred_file, subj_pred)

print("Finished prediction")
