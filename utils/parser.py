from argparse import ArgumentParser

def baseline_arg_parser():
    
    parser = ArgumentParser()

    parser.add_argument("--gpus", type=str,
                        default="",
                        help="Which gpus to use?")
    
    parser.add_argument("--ver",
                        type=str,
                        help="Additional string for the name of the file")
    
    parser.add_argument("--subj_list",
                        type=str,
                        help="File containing the subject ID list, one subject ID on each line")

    parser.add_argument("--rsfc_dir",
                        type=str,
                        help="Directory containing the subject input resting-state functional connectivity files for training and validation")

    parser.add_argument("--mesh_dir",
                        type=str,
                        help="Directory containing the meshes at different resolutions")

    parser.add_argument("--save_dir",
                        type=str,
                        help="Path to the output directory")

    parser.add_argument("--contrast_names",
                        type=str,
                        help="Path to the file containing task contrasts names")

    parser.add_argument("--n_channels_per_hemi",
                        type=int,
                        default=50,
                        help="Number of input channels per hemisphere, default=50")

    parser.add_argument("--n_feat_channels",
                        type=int,
                        default=64,
                        help="Number of input channels per hemisphere, default=50")

    parser.add_argument("--n_output_channels",
                        type=int,
                        default=47,
                        help="Number of output channels per hemisphere, default=47")

    parser.add_argument("--n_samples_per_subj",
                        type=int,
                        default=8,
                        help="Number of rsfc samples per subject, default=8")

    parser.add_argument("--max_mesh_lvl",
                        type=int,
                        default=2,
                        help="The highest mesh resolution level in the model - corresponds to the indices of files under mesh_dir, default=2")

    parser.add_argument("--min_mesh_lvl",
                        type=int,
                        default=0,
                        help="The lowest mesh resolution level in the model - corresponds to the indices of files under mesh_dir, default=2")

    parser.add_argument("--loss",
                        type=str,
                        default="mse",
                        choices=("mse", "rc"),
                        help="Loss function, default: MSE")

    parser.add_argument("--lr",
                        type=float,
                        default=1e-2,
                        help="Learning rate, default: 1e-2")

    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="Training epochs, default: 50")
    
    parser.add_argument("--seed",
                        type=int,
                        default=28,
                        help="Random seed for numpy to create train/val split, default = 28")

    parser.add_argument("--n_val_subj",
                        type=int,
                        default=50,
                        help="Number of subjects in the validation set, default = 50")

    parser.add_argument("--checkpoint_interval",
                        type=int,
                        default=10,
                        help="Number of epochs between saved checkpoints, default = 10")

    parser.add_argument("--batch_size",
                        type=int,
                        default=2,
                        choices={2},
                        help="Only batch of 2 is supported right now")

    parser.add_argument("--init_within_subj_margin",
                        type=float,
                        default=4.0,
                        help="Initial within-subject margin, which should be computed on the training set")
    
    parser.add_argument("--min_within_subj_margin",
                        type=float,
                        default=1.0,
                        help="Minimum within-subject margin that was aimed for")
    
    parser.add_argument("--init_across_subj_margin",
                        type=float,
                        default=5.0,
                        help="Initial across-subject margin, which should be computed on the training set")
    
    parser.add_argument("--max_across_subj_margin",
                        type=float,
                        default=10.0,
                        help="Maximum across-subject margin that was aimed for")

    parser.add_argument("--margin_anneal_step",
                        type=int,
                        default=10,
                        help="Step for annealing the margins")

    return parser

def train_args():
    parser = baseline_arg_parser()

    parser.add_argument("--checkpoint_file",
                        type=str,
                        help="Path to the checkpoint file to be loaded into the model")
    parser.add_argument("--contrast_dir",
                        type=str,
                        help="Directory containing the subject target task contrast files for training and validation")

    return parser.parse_args()

def test_args():
    parser = baseline_arg_parser()

    parser.add_argument("--checkpoint_file",
                        type=str,
                        help="Path to the checkpoint file to be loaded into the model")

    return parser.parse_args()
