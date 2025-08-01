from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import numpy
import torch.nn.functional as F
# from keras.engine.topology import InputLayer


def random_laplace(shape, mu=0., b=1.):
    '''
    Draw random samples from a Laplace distriubtion.

    See: https://en.wikipedia.org/wiki/Laplace_distribution#Generating_random_variables_according_to_the_Laplace_distribution
    '''
    # U = K.random_uniform(shape, -0.5, 0.5)
    U=torch.FloatTensor(shape).uniform_(-0.5, 0.5)
    return mu - b * torch.sign(U) * torch.log(1 - 2 * torch.abs(U))

def random_normal(mean=0.0, std=1.0):
    return torch.normal(mean, std)

def random_multinomial(logits, seed=None):
    '''
    Theano function for sampling from a multinomal with probability given by `logits`
    '''

    return F.one_hot(torch.multinomial(logits, num_samples=1),num_classes=int(logits.shape[1])).squeeze()
    # return torch.multinomial(logits, num_samples=1)

def random_gmm(pi, mu, sig):
    '''
    Sample from a gaussian mixture model. Returns one sample for each row in
    the pi, mu and sig matrices... this is potentially wasteful (because you have to repeat
    the matrices n times if you want to get n samples), but makes it easy to implment
    code where the parameters vary as they are conditioned on different datapoints.
    '''
    # print(mu.shape,sig.shape,'gmm shape')
    normals = random_normal(mu, sig)
    k = random_multinomial(pi)
    # print(k)
    return torch.sum(normals * k, dim=-1, keepdim=True)


