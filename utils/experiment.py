import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from utilities import CONTRASTS, compute_corr_coeff

def train(model, train_loader, optimizer, loss_fn, loss_type, within_subj_margin=0, across_subj_margin=0):
    model.train()
    total_loss = 0
    total_corr = []
    count = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        if loss_type == 'mse':
           loss = loss_fn(output, target)
        elif loss_type == 'rc':
           within_subj_loss, across_subj_loss = loss_fn(output, target)
           loss = torch.clamp(within_subj_loss - within_subj_margin, min=0.0) + torch.clamp(within_subj_loss - across_subj_loss + across_subj_margin, min = 0.0)
        else:
           raise("Invalid loss type")
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
     
        reshaped_output = np.swapaxes(output.cpu().detach().numpy(), 0, 1)
        reshaped_target = np.swapaxes(target.cpu().detach().numpy(), 0, 1)
        corrs = np.diag(compute_corr_coeff(reshaped_output.reshape(reshaped_output.shape[0], -1), reshaped_target.reshape(reshaped_target.shape[0], -1)))
        if batch_idx == 0:
            total_corr = corrs
        else:
            total_corr = total_corr + corrs

        if batch_idx % 50 == 0: 
            print('[{}/{} ({:.0f}%)] Loss: {:.6f} Corr: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), np.mean(corrs)))
    total_loss /= len(train_loader)
    total_corr /= len(train_loader)
    
    print('  Train: avg loss: {:.4f} - avg corr: {:.4f}'.format(total_loss, np.mean(total_corr)))
    for j in range(len(CONTRASTS)):
        print("      %s %s: %.4f, %.4f" % (CONTRASTS[j][0], CONTRASTS[j][1], total_corr[j*2], total_corr[j*2+1]))

    return total_loss, np.mean(total_corr)


def eval(model, val_loader, loss_fn, loss_type, within_subj_margin=0, across_subj_margin=0):
    model.eval()
    total_loss = 0
    total_corr = []
    count = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            n_data = data.size()[0]

            if loss_type == 'mse':
               loss = loss_fn(output, target)
            elif loss_type == 'rc':
               within_subj_loss, across_subj_loss = loss_fn(output, target)
               loss = torch.clamp(within_subj_loss - within_subj_margin, min=0.0) + torch.clamp(within_subj_loss - across_subj_loss + across_subj_margin, min = 0.0)
            else:
               raise("Invalid loss type")
        
            total_loss += loss.item()

            reshaped_output = np.swapaxes(output.cpu().detach().numpy(), 0, 1)
            reshaped_target = np.swapaxes(target.cpu().detach().numpy(), 0, 1)
            corrs = np.diag(compute_corr_coeff(reshaped_output.reshape(reshaped_output.shape[0], -1), reshaped_target.reshape(reshaped_target.shape[0], -1)))
            if batch_idx == 0:
                total_corr = corrs
            else:
                total_corr = total_corr + corrs

    avg_loss = total_loss / len(val_loader)
    avg_corr = total_corr /  len(val_loader)

    print('  Val: avg loss: {:.4f} - avg corr: {:.4f}'.format(avg_loss, np.mean(avg_corr)))
    for j in range(len(CONTRASTS)):
        print("      %s %s: %.4f, %.4f" % (CONTRASTS[j][0], CONTRASTS[j][1], avg_corr[j*2], avg_corr[j*2+1]))

    return avg_loss, np.mean(avg_corr)
