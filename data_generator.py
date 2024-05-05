from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
from torchvision import datasets, transforms
#from causenet.datastream import DataStream, prepare_datastream
from sklearn.preprocessing import OneHotEncoder
import warnings
from itertools import product
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

X_mnist = None
y_mnist = None

def generate_test_demand_design(ood=False,reward=False):
    """
    Returns
    -------
    test_data : TestDataSet
        Uniformly sampled from (p,t,s).
    """
    if reward:
        price=np.array([pmu])
    else:
        price = np.linspace(10, 25, 20)

    if ood:
        time1 = np.linspace(11, 26, 20)
        # time2 = np.linspace(-2, 0, 10)
        # time3 = np.linspace(10, 12, 10)
        # time=np.concatenate([time1,time2,time3])
        time=time1
    else:
        time = np.linspace(0.0, 10, 20)
    emotion_id = np.array([1., 2., 3., 4., 5., 6., 7.])
    emotion = one_hot(emotion_id)

    data = []
    target = []
    treatment=[]
    price = (price - pmu) / psd

    for p, t, s in product(price, time, emotion):
        x=np.concatenate([[t],s])

        # x_latent = np.concatenate([t.reshape((-1, 1)), s.reshape((-1, 1))], axis=1)
        # structural = g(x_latent, None, price)
        # target.append(structural)
        # target.append(storeg(np.expand_dims(x,0),p))
        data.append(x)
        treatment.append([p])


    features = np.array(data)
    # emotion_matrix = one_hot(features[:,2])
    # features=np.concatenate([features[:,:2],emotion_matrix],axis=1)
    treatment=np.array(treatment)

    # targets: np.ndarray = np.array(target)
    targets=storeg(features,treatment)
    print(targets.shape,features.shape,treatment.shape)

    return features,treatment,targets

def get_error(g_hat, g_true, image=False, ood=False):
    seed = np.random.randint(1e9)
    # try:
    #     # test = True ensures we draw test set images
    #     x, z, t, y, g_true = data_fn(ntest, seed,images=has_latent, test=True,ood=ood)
    # except ValueError:
    #     warnings.warn("Too few images, reducing test set size")
    #     ntest = int(ntest * 0.5)
    #     print(ntest)
    #     # test = True ensures we draw test set images
    #     x, z, t, y, g_true = data_fn(ntest, seed,images=has_latent,test=True,ood=ood)

    x,t,y=generate_test_demand_design(ood=ood)
    ## re-draw to get new independent treatment and implied response
    # t = np.linspace(np.percentile(t, 2.5),np.percentile(t, 97.5),ntest).reshape(-1, 1)
    ## we need to make sure z _never_ does anything in these g functions (fitted and true)
    ## above is necesary so that reduced form doesn't win
    # if has_latent:
    #     x_latent, _, _, _, _ = data_fn(ntest, seed, images=False,ood=ood)
    #     y = g_true(x_latent, z, t)
    # else:
    if image:
        pass
    y = g_true(x, t)
    y_true = y.flatten()
    y_hat = g_hat(x, t).flatten().cpu().numpy()
    error=((y_hat - y_true)**2).mean()
    return error


def loadmnist():
    '''
    Load the mnist data once into global variables X_mnist and y_mnist.
    '''
    global X_mnist
    global y_mnist
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    test = datasets.MNIST('../data', train=False,
                       transform=transform)

    X_mnist = []
    y_mnist = []

    for d in [train, test]:
        X = d.data.numpy()
        y = d.targets.numpy()
        # X, y = d
        X = X.astype('float')
        idx = np.argsort(y)
        # print(X.shape,y.shape)
        X_mnist.append(X[idx, :, :])
        y_mnist.append(y[idx])

def get_images(digit, n, seed=None, testset=False):
    if X_mnist is None:
        loadmnist()
    is_test = int(testset)
    rng = np.random.RandomState(seed)
    X_i = X_mnist[is_test][y_mnist[is_test] == digit, :, :]
    n_i, i, j = X_i.shape
    perm = rng.permutation(np.arange(n_i))
    if n > n_i:
        raise ValueError('You requested %d images of digit %d when there are \
						  only %d unique images in the %s set.' % (n, digit, n_i, 'test' if testset else 'training'))
    return X_i[perm[0:n], :, :].reshape((n,i*j))

def one_hot(col, **kwargs):
    z = col.reshape(-1,1)
    enc = OneHotEncoder(sparse=False, **kwargs)
    return enc.fit_transform(z)

def get_test_valid_train(generator, n, batch_size=128, seed=123, **kwargs):
    x, z, t, y, g = generator(n=int(n*0.6), seed=seed, **kwargs)
    train = prepare_datastream(x, z, t, y, True, batch_size, **kwargs)
    x, z, t, y, g = generator(n=int(n*0.2), seed=seed+1, **kwargs)
    valid = prepare_datastream(x, z, t, y, False, batch_size, **kwargs)
    x, z, t, y, g = generator(n=int(n*0.2), seed=seed+2, **kwargs)
    test = prepare_datastream(x, z, t, y, False, batch_size, **kwargs)
    return train, valid, test, g

def sensf(x):
    return 2.0*((x - 5)**4 / 600 + np.exp(-((x - 5)/0.5)**2) + x/10. - 2)

def emocoef(emo):
    emoc = (emo * np.array([1., 2., 3., 4., 5., 6., 7.])[None, :]).sum(axis=1)
    return emoc

psd = 3.7
pmu = 17.779
ysd = 158.#292.
ymu = -292.1

def storeg(x, price):
    emoc = emocoef(x[:, 1:])
    time = x[:, 0]
    # print(time.shape,emoc.shape)
    g = sensf(time)*emoc*10. + (emoc*sensf(time)-2.0)*(psd*price.flatten() + pmu)
    y = (g - ymu)/ysd
    # print(y.shape)
    return y.reshape(-1, 1)

def deep_IV_data(n, seed=1, ynoise=1., pnoise=1., ypcor=0.8, ood=False,use_images=False, test=False):
    rng = np.random.RandomState(seed)

    # covariates: time and emotion
    time = rng.rand(n) * 10
    if ood:
        time= rng.rand(n) * 3
        time+=3.5
    emotion_id = rng.randint(0, 7, size=n)
    emotion = one_hot(emotion_id)
    if use_images:
        idx = np.argsort(emotion_id)
        emotion_feature = np.zeros((0, 28*28))
        for i in range(7):
            img = get_images(i, np.sum(emotion_id == i), seed, test)
            emotion_feature = np.vstack([emotion_feature, img])
        reorder = np.argsort(idx)
        emotion_feature = emotion_feature[reorder, :]
    else:
        emotion_feature = emotion

    # random instrument
    z = rng.randn(n)

    # z -> price
    v = rng.randn(n)*pnoise
    price = sensf(time)*(z + 3)  + 25.
    price = price + v
    price = (price - pmu)/psd

    # true observable demand function
    x = np.concatenate([time.reshape((-1, 1)), emotion_feature], axis=1)
    x_latent = np.concatenate([time.reshape((-1, 1)), emotion], axis=1)
    g = lambda x, p: storeg(x, p) # doesn't use z

    # errors 
    e = (ypcor*ynoise/pnoise)*v + rng.randn(n)*ynoise*np.sqrt(1-ypcor**2)
    e = e.reshape(-1, 1)
    
    # response
    y = g(x_latent, price) + e

    return (x,
            z.reshape((-1, 1)),
            price.reshape((-1, 1)),
            y.reshape((-1, 1)),
            g)


def linear_data(n, seed=None, sig_d=0.5, sig_y=2, sig_t=1.5,
				alpha=4, noiseless_t=False, **kwargs):
    rng = np.random.RandomState(seed)
    nox = lambda z, d: z + 2*d
    house_price = lambda alpha, d, nox_val: alpha + 4*d + 2*nox_val

    d = rng.randn(n) * sig_d
    law = rng.randint(0, 2, n)

    if noiseless_t:
        t = nox(law, d.mean()) + sig_t*rng.randn(n)
    else:
        t = (nox(law, d) + sig_t*rng.randn(n) - 0.5) / 1.8
    z = law.reshape((-1, 1))
    x = np.zeros((n, 0))
    y = (house_price(alpha, d, t) + sig_y*rng.randn(n) - 5.)/5.
    g_true = lambda x, z, t: house_price(alpha, 0, t)
    return x, z, t.reshape((-1, 1)), y.reshape((-1, 1)), g_true


def get_normal_params(mV, mX, mU=1, depX=0.0, depU=0.1):
    m = mV + mX + mU
    mu = np.zeros(m)

    sig = np.eye(m)
    temp_sig = np.ones(shape=(m - mV, m - mV))
    temp_sig = temp_sig * depU
    sig[mV:, mV:] = temp_sig

    sig_temp = np.ones(shape=(mX, mX)) * depX
    sig[mV:-mU, mV:-mU] = sig_temp

    sig[np.diag_indices_from(sig)] = 1

    return mu, sig


class Gen_fn_IVCluster(object):
    def __init__(self) -> None:
        self.config = {
            'data': 'fn_IVCluster',
            'reps': 10,
            'seed': 2022,
            'fn': '2dpoly',
            'num': 3000,
            'numDomain': 3,
            'x_dim': 3,
            'u_coef': 2,
            'x_fn': 'linear',
            'y_fn': 'n',
            'x4u': 0.1,
            'dataDir': './Data/data/fn_IVCluster/2dpoly/3000_3_3_2_linear_n_0.1/',
        }

    def set_Configuration(self, config=None):
        if config is not None:
            self.config = config

        self.config['dataDir'] = './Data/data/fn_IVCluster/{}/{}_{}_{}_{}_{}_{}_{}/'.format(self.config['fn'],
                                                                                            self.config['num'],
                                                                                            self.config['numDomain'],
                                                                                            self.config['x_dim'],
                                                                                            self.config['u_coef'],
                                                                                            self.config['x_fn'],
                                                                                            self.config['y_fn'],
                                                                                            self.config['x4u'])

    def initiation(self, G=False):
        self.num = self.config['num']
        self.reps = self.config['reps']
        self.seed = self.config['seed']
        self.fn = self.config['fn']
        self.dataDir = self.config['dataDir']
        self.numDomain = self.config['numDomain']
        self.x_dim = self.config['x_dim']
        self.u_coef = self.config['u_coef']
        self.x_fn = self.config['x_fn']
        self.y_fn = self.config['y_fn']
        self.x4u = self.config['x4u']

        set_seed(667)
        self.x_coef = np.array([random.choices(np.arange(-1, 1, 0.1).round(2), k=10) for _ in range(5)])
        self.x_coef[:, 0] = np.array([0.8, 0.2, -0.8, -0.2, -1.0])

        self.fn_xt = lambda coef, x: np.sum([coef[i] * x[:, i] for i in range(self.x_dim)], 0).reshape(-1, 1)

        set_seed(self.seed)
        if self.x_fn == 'IHDP' or self.x_fn == 'PM25':
            Data22 = realData(dataName=self.x_fn)
            self.Data22 = Data22
        if not os.path.exists(self.dataDir + '/1/train.csv') or G:
            print('Next, run dataGenerator: ')
            for rep_i in range(self.config['reps']):
                self.mean = None
                self.std = None
                self.gen_exp(exp=rep_i, save=True)
            print('-' * 30)

    def true_g_function_np(self, x):
        func = self.fn
        if func == 'abs':
            return np.abs(x)
        elif func == '2dpoly':
            return -1.5 * x + .9 * (x ** 2)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-1 * x))
        elif func == 'sin':
            return np.sin(x)
        elif func == 'cos':
            return np.cos(x)
        elif func == 'step':
            return 1. * (x < 0) + 2.5 * (x >= 0)
        elif func == '3dpoly':
            return -1.5 * x + .9 * (x ** 2) + x ** 3
        elif func == 'linear':
            return x
        elif func == 'rand_pw':
            pw_linear = self._generate_random_pw_linear()
            return np.reshape(np.array([pw_linear(x_i) for x_i in x.flatten()]), x.shape)
        else:
            raise NotImplementedError()

    def backF(self, x, func='linear'):
        if func == 'i' or func == 'identity':
            return x
        elif func == 'abs':
            return x + np.abs(x)
        elif func == 'poly':
            return x + (x ** 2)
        elif func == 'sigmoid':
            return x + 1 / (1 + np.exp(-1 * x))
        elif func == 'sin':
            return x + np.sin(x)
        elif func == 'cos':
            return x + np.cos(x)
        elif func == 'linear':
            return x + 0
        elif func == 'rand_pw':
            pw_linear = self._generate_random_pw_linear()
            return np.reshape(np.array([pw_linear(x_i) for x_i in x.flatten()]), x.shape)
        else:
            print("The data x is from : {}".format(self.x_fn))
            return x

    def normalize(self, y):
        return (y - self.mean) / self.std

    def denormalize(self, y):
        return y * self.std + self.mean

    def gen_t0(self, t, x, u, e2):

        g = self.true_g_function_np(t - t)
        y = g + 2 * np.sum(x, 1, keepdims=True) / self.x_dim + self.u_coef * u + e2
        v = g + 2 * np.sum(x, 1, keepdims=True) / self.x_dim

        if self.y_fn == 'n' or self.y_fn == 'nonlinear' or self.y_fn == 'non':
            y = y - np.abs(x[:, 0:1] * x[:, 1:2]) - np.sin(10 + x[:, 2:3] * x[:, 2:3])
            v = v - np.abs(x[:, 0:1] * x[:, 1:2]) - np.sin(10 + x[:, 2:3] * x[:, 2:3])

        y = self.normalize(y)
        g = self.normalize(g)
        v = self.normalize(v)

        return cat([g, v, y])

    def gen_data(self, num, mode='train'):
        if self.x_fn == 'IHDP' or self.x_fn == 'PM25':
            if mode == 'train':
                num = self.Data22.x_train.shape[0]
            elif mode == 'valid':
                num = self.Data22.x_valid.shape[0]
            elif mode == 'test':
                num = self.Data22.x_test.shape[0]

        mu, sig = get_normal_params(0, self.x_dim, 1, 0, self.x4u)
        temp = np.random.multivariate_normal(mean=mu, cov=sig, size=num)

        x = temp[:, :self.x_dim]
        u = temp[:, self.x_dim:]
        z = np.random.choice(list(range(0, self.numDomain)), (num, 1))
        e1 = np.random.normal(0, .1, size=(num, 1))
        e2 = np.random.normal(0, .1, size=(num, 1))

        if self.x_fn == 'IHDP' or self.x_fn == 'PM25':
            if mode == 'train':
                x = self.Data22.x_train
            elif mode == 'valid':
                x = self.Data22.x_valid
            elif mode == 'test':
                x = self.Data22.x_test

        x = x[:num, :self.x_dim]

        x_fn = self.backF(x, self.x_fn)

        if self.x_fn == 'UE':
            t_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.2 * u),
                            (self.fn_xt(self.x_coef[1], x_fn) + 0.2 * u),
                            (self.fn_xt(self.x_coef[2], x_fn) + 0.2 * u),
                            (self.fn_xt(self.x_coef[3], x_fn) + 0.2 * u),
                            (self.fn_xt(self.x_coef[4], x_fn) + 0.2 * u)], 1)
            d_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.2 * 0),
                            (self.fn_xt(self.x_coef[1], x_fn) + 0.2 * 0),
                            (self.fn_xt(self.x_coef[2], x_fn) + 0.2 * 0),
                            (self.fn_xt(self.x_coef[3], x_fn) + 0.2 * 0),
                            (self.fn_xt(self.x_coef[4], x_fn) + 0.2 * 0)], 1)
        elif self.x_fn == 'UV':
            t_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.5 * u),
                            (self.fn_xt(self.x_coef[1], x_fn) + 0.5 * u + 1),
                            (self.fn_xt(self.x_coef[2], x_fn) + 0.5 * u),
                            (self.fn_xt(self.x_coef[3], x_fn) + 0.5 * u - 0.2),
                            (self.fn_xt(self.x_coef[4], x_fn) + 0.5 * u - 0.8)], 1)
            d_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.5 * 0),
                            (self.fn_xt(self.x_coef[1], x_fn) + 0.5 * 0 + 1),
                            (self.fn_xt(self.x_coef[2], x_fn) + 0.5 * 0),
                            (self.fn_xt(self.x_coef[3], x_fn) + 0.5 * 0 - 0.2),
                            (self.fn_xt(self.x_coef[4], x_fn) + 0.5 * 0 - 0.8)], 1)
        elif self.x_fn == 'UEV':
            t_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.2 * u),
                            (self.fn_xt(self.x_coef[1], x_fn) - 0.5 * u),
                            (self.fn_xt(self.x_coef[2], x_fn) + 0.4 * u),
                            (self.fn_xt(self.x_coef[3], x_fn) - 0.2 * u),
                            (self.fn_xt(self.x_coef[4], x_fn) + 0.1 * u)], 1)
            d_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.2 * 0),
                            (self.fn_xt(self.x_coef[1], x_fn) - 0.5 * 0),
                            (self.fn_xt(self.x_coef[2], x_fn) + 0.4 * 0),
                            (self.fn_xt(self.x_coef[3], x_fn) - 0.2 * 0),
                            (self.fn_xt(self.x_coef[4], x_fn) + 0.1 * 0)], 1)
        else:
            t_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.2 * u),
                            (self.fn_xt(self.x_coef[1], x_fn) + 0.2 * u + 1),
                            (self.fn_xt(self.x_coef[2], x_fn) + 0.2 * u),
                            (self.fn_xt(self.x_coef[3], x_fn) + 0.2 * u - 0.2),
                            (self.fn_xt(self.x_coef[4], x_fn) + 0.2 * u - 0.8)], 1)
            d_matrix = cat([(self.fn_xt(self.x_coef[0], x_fn) + 0.2 * 0),
                            (self.fn_xt(self.x_coef[1], x_fn) + 0.2 * 0 + 1),
                            (self.fn_xt(self.x_coef[2], x_fn) + 0.2 * 0),
                            (self.fn_xt(self.x_coef[3], x_fn) + 0.2 * 0 - 0.2),
                            (self.fn_xt(self.x_coef[4], x_fn) + 0.2 * 0 - 0.8)], 1)
        t = np.array([t_matrix[i][z_i] for i, z_i in enumerate(z)]) + e1
        d = np.array([d_matrix[i][z_i] for i, z_i in enumerate(z)])

        g = self.true_g_function_np(t)
        y = g + 2 * np.sum(x, 1, keepdims=True) / self.x_dim + self.u_coef * u + e2
        v = g + 2 * np.sum(x, 1, keepdims=True) / self.x_dim

        if self.y_fn == 'n' or self.y_fn == 'nonlinear' or self.y_fn == 'non':
            y = y - np.abs(x[:, 0:1] * x[:, 1:2]) - np.sin(10 + x[:, 2:3] * x[:, 2:3])
            v = v - np.abs(x[:, 0:1] * x[:, 1:2]) - np.sin(10 + x[:, 2:3] * x[:, 2:3])

        if self.mean is None:
            self.mean = y.mean()
            self.std = y.std()

        y = self.normalize(y)
        g = self.normalize(g)
        v = self.normalize(v)

        m = self.gen_t0(t, x, u, e2)

        data_df = pd.DataFrame(np.concatenate([x, u, z, t, d, y, g, v, m, t], 1),
                               columns=['x{}'.format(i + 1) for i in range(x.shape[1])] +
                                       ['u{}'.format(i + 1) for i in range(u.shape[1])] +
                                       ['z{}'.format(i + 1) for i in range(z.shape[1])] +
                                       ['t{}'.format(i + 1) for i in range(t.shape[1])] +
                                       ['d{}'.format(i + 1) for i in range(d.shape[1])] +
                                       ['y{}'.format(i + 1) for i in range(y.shape[1])] +
                                       ['g{}'.format(i + 1) for i in range(g.shape[1])] +
                                       ['v{}'.format(i + 1) for i in range(v.shape[1])] +
                                       ['m{}'.format(i + 1) for i in range(m.shape[1])] +
                                       ['w{}'.format(i + 1) for i in range(t.shape[1])])

        return data_df


    def ihdp_g(self,x,t):
        g = self.true_g_function_np(t)
        v = g + 2 * np.sum(x, 1, keepdims=True) / x.shape[-1]

        v = v - np.abs(x[:, 0:1] * x[:, 1:2]) - np.sin(10 + x[:, 2:3] * x[:, 2:3])
        v = self.normalize(v)
        return v

    def ground_truth(self, x, t, u=None):
        if u is None:
            return self.normalize(self.true_g_function_np(t)), self.normalize(
                self.true_g_function_np(t) + 2 * x), self.normalize(self.true_g_function_np(t) + 2 * x)
        else:
            return self.normalize(self.true_g_function_np(t)), self.normalize(
                self.true_g_function_np(t) + 2 * x), self.normalize(self.true_g_function_np(t) + 2 * x + 2 * u)

    def gen_exp(self, exp=1, save=False):
        np.random.seed(exp * 527 + self.seed)
        print(f'Generate Causal Cluster datasets - {exp}/{self.reps}. ')

        if self.x_fn == 'IHDP' or self.x_fn == 'PM25':
            self.Data22.shuffle()
        self.train_df = self.gen_data(self.num, 'train')
        self.valid_df = self.gen_data(self.num, 'valid')
        self.test_df = self.gen_data(self.num, 'test')

        if save:
            data_path = self.dataDir + '/{}/'.format(exp)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)

            self.train_df.to_csv(data_path + '/train.csv', index=False)
            self.valid_df.to_csv(data_path + '/val.csv', index=False)
            self.test_df.to_csv(data_path + '/test.csv', index=False)

            np.savez(data_path + '/mean_std.npz', mean=self.mean, std=self.std)
            np.savez(data_path + '/coefs.npz', x_coef=self.x_coef)

        train = CausalDataset(self.train_df, variables=['x', 'u', 'z', 't', 'd', 'y', 'g', 'v', 'm', 'w', 'c'])
        valid = CausalDataset(self.valid_df, variables=['x', 'u', 'z', 't', 'd', 'y', 'g', 'v', 'm', 'w', 'c'])
        test = CausalDataset(self.test_df, variables=['x', 'u', 'z', 't', 'd', 'y', 'g', 'v', 'm', 'w', 'c'])

        return Data(train, valid, test, self.num)

    def get_exp(self, exp, num=0):
        subDir = self.dataDir + f'/{exp}/'

        self.train_df = pd.read_csv(subDir + 'train.csv')
        self.val_df = pd.read_csv(subDir + 'val.csv')
        self.test_df = pd.read_csv(subDir + 'test.csv')

        if not (num > 0 and num < len(self.train_df)):
            num = len(self.train_df)

        train = CausalDataset(self.train_df[:num], variables=['x', 'u', 'z', 't', 'd', 'y', 'g', 'v', 'm', 'w', 'c'])
        val = CausalDataset(self.val_df[:num], variables=['x', 'u', 'z', 't', 'd', 'y', 'g', 'v', 'm', 'w', 'c'])
        test = CausalDataset(self.test_df[:num], variables=['x', 'u', 'z', 't', 'd', 'y', 'g', 'v', 'm', 'w', 'c'])

        mean_std = np.load(subDir + '/mean_std.npz', allow_pickle=True)
        self.mean = mean_std['mean'].reshape(1)[0]
        self.std = mean_std['std'].reshape(1)[0]

        coefs = np.load(subDir + '/coefs.npz', allow_pickle=True)
        self.x_coef = coefs['x_coef']

        return Data(train, val, test, num)



def get_var_df(df, var):
    var_cols = [c for c in df.columns if c.startswith(var)]
    return df[var_cols].to_numpy()


def cat(data_list, axis=1):
    try:
        output = torch.cat(data_list, axis)
    except:
        output = np.concatenate(data_list, axis)

    return output


class Data(object):
    def __init__(self, train, valid, test, num):
        self.train = train
        self.valid = valid
        self.test = test
        self.num = num

    def transform(self, load_dict):
        if load_dict['type'] == 'numpy' or load_dict['type'] == 'np':
            try:
                self.numpy()
            except:
                pass

            return self.train

        if load_dict['type'] == 'tensor':
            try:
                self.tensor()
            except:
                pass

        if load_dict['type'] == 'double':
            try:
                self.double()
            except:
                pass

        if load_dict['GPU']:
            try:
                self.cuda()
            except:
                pass

        loader = self.get_loader(load_dict)
        return loader

    def tensor(self):
        try:
            self.train.to_tensor()
            self.valid.to_tensor()
            self.test.to_tensor()
        except:
            pass

    def double(self):
        try:
            self.train.to_double()
            self.valid.to_double()
            self.test.to_double()
        except:
            pass

    def cpu(self):
        try:
            self.train.to_cpu()
            self.valid.to_cpu()
            self.test.to_cpu()
        except:
            pass

    def detach(self):
        try:
            self.train.detach()
            self.valid.detach()
            self.test.detach()
        except:
            pass

    def numpy(self):
        try:
            self.train.to_numpy()
            self.valid.to_numpy()
            self.test.to_numpy()
        except:
            pass

    def cuda(self, n=0, type='float'):
        if type == 'float':
            try:
                self.tensor()
            except:
                pass
        elif type == 'double':
            try:
                self.double()
            except:
                pass

        try:
            self.train.to_cuda(n)
            self.valid.to_cuda(n)
            self.test.to_cuda(n)
        except:
            pass

    def get_loader(self, load_dict, data=None):
        if data is None:
            data = self.train
        loader = DataLoader(data, batch_size=load_dict['batch_size'])
        return loader

    def split(self, split_ratio=0.5, data=None):
        if data is None: data = self.train
        self.data1 = copy.deepcopy(data)
        self.data2 = copy.deepcopy(data)

        split_num = int(data.length * split_ratio)
        self.data1.split(0, split_num)
        self.data2.split(split_num, data.length)

        return self.data1, self.data2


class CausalDataset(Dataset):
    def __init__(self, df, variables=['u', 'x', 'v', 'z', 'p', 'm', 't', 'y', 'f', 'c'], observe_vars=['z', 't']):
        if not 'c' in variables: variables.append('c')

        self.length = len(df)
        self.variables = variables

        for var in variables:
            exec(f'self.{var}=get_var_df(df, \'{var}\')')

        observe_list = []
        for item in observe_vars:
            exec(f'observe_list.append(self.{item})')
        self.c = np.concatenate(observe_list, axis=1)

    def to_cpu(self):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.cpu()')

    def to_cuda(self, n=0):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.cuda({n})')

    def to_tensor(self):
        if type(self.t) is np.ndarray:
            for var in self.variables:
                exec(f'self.{var} = torch.Tensor(self.{var})')
        else:
            for var in self.variables:
                exec(f'self.{var} = self.{var}.float()')

    def to_double(self):
        if type(self.t) is np.ndarray:
            for var in self.variables:
                exec(f'self.{var} = torch.Tensor(self.{var}).double()')
        else:
            for var in self.variables:
                exec(f'self.{var} = self.{var}.double()')

    def to_numpy(self):
        try:
            self.detach()
        except:
            pass
        try:
            self.to_cpu()
        except:
            pass

        for var in self.variables:
            exec(f'self.{var} = self.{var}.numpy()')

    def shuffle(self):
        idx = list(range(self.length))
        random.shuffle(idx)
        for var in self.variables:
            try:
                exec(f'self.{var} = self.{var}[idx]')
            except:
                pass

    def split(self, start, end):
        for var in self.variables:
            try:
                exec(f'self.{var} = self.{var}[start:end]')
            except:
                pass

        self.length = end - start

    def to_pandas(self):
        var_list = []
        var_dims = []
        var_name = []
        for var in self.variables:
            exec(f'var_list.append(self.{var})')
            exec(f'var_dims.append(self.{var}.shape[1])')
        for i in range(len(self.variables)):
            for d in range(var_dims[i]):
                var_name.append(self.variables[i] + str(d))
        df = pd.DataFrame(np.concatenate(var_list, axis=1), columns=var_name)
        return df

    def detach(self):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.detach()')

    def __getitem__(self, idx):
        var_dict = {}
        for var in self.variables:
            exec(f'var_dict[\'{var}\']=self.{var}[idx]')

        return var_dict

    def __len__(self):
        return self.length

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
