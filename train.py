import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

from model.brain_surf_cnn import BrainSurfCNN
from utils import experiment
from utils.parser import train_args
from utils.dataset import MultipleSampleMeshDataset
from utils.utilities import CONTRASTS, parse_contrasts_names, save_checkpoint, contrast_mse_loss

if __name__ == "__main__":

    args = train_args()

    """Init"""
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus

    output_name = "%s_feat%d_s%d_c%d_lr%s_seed%d" % (args.ver, args.n_feat_channels, args.n_samples_per_subj, args.n_channels_per_hemi, str(args.lr), args.seed)
    output_dir = os.path.join(args.save_dir, output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise Exception("Output dir exists: %s" % output_dir)


    """Load Data"""
    np.random.seed(args.seed)
    subj_ids = np.genfromtxt(args.subj_list, dtype="<U13")
    np.random.shuffle(subj_ids)
    val_subj_ids = subj_ids[:args.n_val_subj]
    train_subj_ids = subj_ids[args.n_val_subj:]

    train_dataset = MultipleSampleMeshDataset(
        train_subj_ids,
        rsfc_dir=args.rsfc_dir,
        contrast_dir=args.contrast_dir,
        num_samples=args.n_samples_per_subj)
    val_dataset = MultipleSampleMeshDataset(
        val_subj_ids,
        rsfc_dir=args.rsfc_dir,
        contrast_dir=args.contrast_dir,
        num_samples=args.n_samples_per_subj)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    print("Number of training subjects:", len(train_loader))
    print("Number of validation subjects:", len(val_loader))

    contrasts = CONTRASTS if args.contrast_names is None else parse_contrasts_names(args.contrast_names)

    """Init model"""
    """two hemispheres are concatenated"""
    model = BrainSurfCNN(
        mesh_dir=args.mesh_dir,
        in_ch=args.n_channels_per_hemi*2,
        out_ch=args.n_output_channels*2,
        max_level=args.max_mesh_lvl,
        min_level=args.min_mesh_lvl,
        fdim=args.n_feat_channels)
    model.cuda()

    """Loading checkpoint"""
    if args.checkpoint_file is not None:
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


    """Optimizer"""
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if (args.loss == "mse"):
        loss_fn = nn.MSELoss()
    elif (args.loss == "rc"):
        loss_fn = contrast_mse_loss
    else:
        print("Loss is not implemented")

    """Cooking"""
    val_losses = []
    val_corrs = []
    best_loss = sys.float_info.max
    best_corr = 0.0

    for epoch in range(args.epochs):
        if args.loss == 'rc':
            within_subj_margin = np.max((args.init_within_subj_margin * 0.5**(epoch // args.margin_anneal_step), args.min_within_subj_margin))
            across_subj_margin = np.min((args.init_across_subj_margin * 2**(epoch // args.margin_anneal_step), args.max_across_subj_margin))
            print("Epoch %d: within-subject margin %.1f - across-subj margin %.1f" % (epoch, within_subj_margin, across_subj_margin))
        else:
            within_subj_margin = 0
            across_subj_margin = 0
            print("Epoch %d" % (epoch + 1))

        train_loss, train_corr = experiment.train(model, train_loader, contrasts, optimizer, loss_fn, args.loss, within_subj_margin, across_subj_margin)
        val_loss, val_corr     = experiment.eval(model, val_loader, contrasts, loss_fn, args.loss, within_subj_margin, across_subj_margin)

        val_losses.append(val_loss)
        val_corrs.append(val_corr)

        mean_loss = np.mean(val_losses[-args.checkpoint_interval:])
        mean_corr = np.mean(val_corrs[-args.checkpoint_interval:])
        if mean_loss < best_loss:
            save_checkpoint(model, optimizer, scheduler, epoch, "best_loss.pth", output_dir)
            best_loss = mean_loss
        if mean_corr > best_corr:
            save_checkpoint(model, optimizer, scheduler, epoch, "best_corr.pth", output_dir)
            best_corr = mean_corr

        if (epoch % args.checkpoint_interval == 0):
            save_checkpoint(model, optimizer, scheduler, epoch, "checkpoint_%d.pth" % epoch, output_dir)
        scheduler.step()
    save_checkpoint(model, optimizer, scheduler, args.epochs, "checkpoint_%d.pth" % args.epochs, output_dir)
