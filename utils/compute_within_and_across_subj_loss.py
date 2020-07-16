import os
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--subj_list",
                    type=str,
                    help="File containing the subject ID list, one subject ID on each line")

parser.add_argument("--pred_dir",
                    type=str,
                    help="Path to the output directory")

parser.add_argument("--contrast_dir",
                    type=str,
                    help="Directory containing the subject target task contrast files for training and validation")

parser.add_argument("--mask",
                    type=str,
                    help="File containing medial wall mask")

parser.add_argument("--batch_size",
                    type=int,
                    default=10,
                    help="Number of subjects used to compute the within/across subject loss")

def compute_loss(pred, target, mask):
    avg_pred = np.mean(pred, 0)
    lh_pred = avg_pred[::2, mask[0, :]]
    rh_pred = avg_pred[1::2, mask[1, :]]

    lh_target = target[::2, mask[0, :]]
    rh_target = target[1::2, mask[1, :]]

    flat_pred = np.concatenate((lh_pred.flatten(), rh_pred.flatten()), axis=0)
    flat_target = np.concatenate((lh_target.flatten(), rh_target.flatten()), axis=0)

    return np.mean((flat_pred - flat_target)**2)

if __name__ == "__main__":

    args = parser.parse_args()

    subj_ids = np.genfromtxt(args.subj_list, dtype="<U13")
    mask = np.load(args.mask)

    within_mse = 0
    across_mse = 0
    for i in range(len(subj_ids)):
        print(i+1, "/", len(subj_ids))
        subj = subj_ids[i]
        samples = np.random.choice(subj_ids, args.batch_size+1, replace=False)
        other_samples = np.setdiff1d(samples, np.asarray([subj]))[:args.batch_size]

        pred = np.load(os.path.join(args.pred_dir, "%s_pred.npy" % subj))
        target = np.load(os.path.join(args.contrast_dir, "%s_joint_LR_task_contrasts.npy" % subj))

        within_mse += (compute_loss(pred, target, mask) / len(subj_ids))

        for s in other_samples:
            other_target = np.load(os.path.join(args.contrast_dir, "%s_joint_LR_task_contrasts.npy" % s))
            across_mse += compute_loss(pred, other_target, mask) / (len(other_samples) * len(subj_ids))

    print("Within MSE", within_mse)
    print("Across MSE", across_mse)
    np.save("within_and_across_mse.npy", { "within": within_mse, "across": across_mse })
