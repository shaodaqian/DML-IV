from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import numpy as np
import torch.nn.functional as F
from deepiv.utils import device



def log_norm_pdf(x, mu, log_sig):
    z = (x - mu) / (torch.exp(torch.clamp(log_sig, -40, 40))) #TODO: get rid of this clipping
    return -(0.5)*np.log(2*np.pi) - log_sig - 0.5*((z)**2)


# def mixture_of_gaussian_output(x, n_components):
#     mu = keras.layers.Dense(n_components, activation='linear')(x)
#     log_sig = keras.layers.Dense(n_components, activation='linear')(x)
#     pi = keras.layers.Dense(n_components, activation='softmax')(x)
#     return Concatenate(axis=1)([pi, mu, log_sig])

def mixture_of_gaussian_loss(y_true, out):
    '''
    Combine the mixture of gaussian distribution and the loss into a single function
    so that we can do the log sum exp trick for numerical stability...
    '''
    [pi, mu, log_sig] = out

    gauss = log_norm_pdf(torch.repeat_interleave(y_true, mu.shape[1], dim=1), mu, log_sig)
    # TODO: get rid of clipping.
    gauss = torch.clamp(gauss, -40, 40)
    max_gauss = max(0, torch.max(gauss))
    # log sum exp trick...
    gauss = gauss - max_gauss
    out = torch.sum(pi * torch.exp(gauss), axis=1)
    loss = torch.mean(-torch.log(out) + max_gauss)
    return loss
