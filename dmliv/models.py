from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
import time
from .samplers import random_gmm
from .densities import mixture_of_gaussian_loss
from .architectures import FeedForward,MixtureGaussian,ConvNet
from .utils import device,Averager,ResponseDataset

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from scipy.stats import norm
from data_generator import sensf,psd,pmu,storeg

def train_Ymodel(Ymodel,train_data,valid_data,batch_size,epochs,lr,w_decay):

    # train_data = ResponseDataset(z, x, t,y)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        shuffle=False,  # 'True' to check training progress with validation function.
        num_workers=0,
        pin_memory=True)

    # data parallel for multi-GPU
    # model = torch.nn.DataParallel(model,device_ids=[0,1,2]).to(device)
    Ymodel = Ymodel.to(device)
    Ymodel.train()
    print("Model:")
    print(Ymodel)

    """ setup loss """
    criterion = nn.MSELoss()
    # loss averager

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, Ymodel.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    optimizer = optim.AdamW(filtered_parameters, lr=lr,weight_decay=w_decay)
    # optimizer = optim.SGD(filtered_parameters, lr=0.01,weight_decay=w_decay)

    print("Optimizer:")
    print(optimizer)
    loss_avg = Averager()

    valid_x=torch.tensor(valid_data[0]).to(device)
    valid_z=torch.tensor(valid_data[1]).to(device)
    valid_y=torch.tensor(valid_data[3]).to(device)

    start_time = time.time()
    for ep in range(epochs):
        # train part
        for i, (inst, feat, treat,resp) in enumerate(train_loader):
            inst = inst.to(device)
            feat = feat.to(device)
            resp = resp.float().to(device)

            out = Ymodel(inst, feat)
            cost = criterion(resp, out)
            # print(resp,out,cost.item())

            Ymodel.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(Ymodel.parameters(), 5)  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)
            # print(cost.item())
            # validation part

        if (ep + 1) % 100 == 0:  # To see training progress, we also conduct validation when 'iteration == 0'
            elapsed_time = time.time() - start_time
            Ymodel.eval()
            valid_out=Ymodel(valid_z,valid_x)
            valid_loss=criterion(valid_y,valid_out)
            # training loss and validation loss
            loss_log = f'[{ep + 1}/{epochs}] Train: {loss_avg.val():0.3f}, Valid: {valid_loss.item():0.3f}'
            loss_avg.reset()
            print(loss_log)
            Ymodel.train()
    return Ymodel

class Ymodel(nn.Module):
    '''
    Adds sampling functionality to a Keras model and extends the losses to support
    mixture of gaussian losses.

    # Argument
    '''

    def __init__(self, input_channel, hiddens,dropout,use_image=False,image_shape=None):
        super(Ymodel, self).__init__()
        self.use_image=use_image
        self.image_shape=image_shape
        if use_image:
            self.net=ConvNet(hiddens,dropout=dropout)
        else:
            self.net=FeedForward(input_channel,hiddens,dropout)
        self.output=nn.Linear(hiddens[2], 1)

    def forward(self,inst,feat):
        if self.use_image:
            time,image=feat[:, 0:1], feat[:, 1:]
            image=image.reshape(self.image_shape)
            fnn = self.net(image,[time,inst])
        else:
            inp=torch.cat((inst,feat),-1).float()
            fnn=self.net(inp)
        out=self.output(fnn)
        return out


def train_treatment(treatment_model,train_data,valid_data,batch_size,loss,epochs,lr,w_decay):

    # train_data = TreatmentDataset(z, x, t)
    # train_data = ResponseDataset(z,x,t,y)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=0,
        pin_memory=True)

    # data parallel for multi-GPU
    # model = torch.nn.DataParallel(model,device_ids=[0,1,2]).to(device)
    treatment_model = treatment_model.to(device)
    treatment_model.train()
    print("Model:")
    print(treatment_model)

    """ setup loss """
    if loss == 'mixture_of_gaussians':
        criterion = mixture_of_gaussian_loss
    # loss averager

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, treatment_model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    optimizer = optim.AdamW(filtered_parameters, lr=lr,weight_decay=w_decay)
    # optimizer = optim.SGD(filtered_parameters, lr=0.01,weight_decay=w_decay)

    print("Optimizer:")
    print(optimizer)
    loss_avg = Averager()

    start_time = time.time()
    valid_x=torch.tensor(valid_data[0]).to(device)
    valid_z=torch.tensor(valid_data[1]).to(device)
    valid_t=torch.tensor(valid_data[2]).to(device)
    for ep in range(epochs):
        # train part
        for i, (inst, feat, treat,resp) in enumerate(train_loader):
            inst = inst.to(device)
            feat = feat.to(device)
            treat = treat.to(device)

            out = treatment_model(inst, feat)
            cost = criterion(treat, out)

            treatment_model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(treatment_model.parameters(), 5)  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)
            # print(cost.item())
            # validation part

        if (ep + 1) % 100 == 0:  # To see training progress, we also conduct validation when 'iteration == 0'
            elapsed_time = time.time() - start_time
            treatment_model.eval()
            valid_out=treatment_model(valid_z,valid_x)
            valid_loss=criterion(valid_t,valid_out)
            # training loss and validation loss
            loss_log = f'[{ep + 1}/{epochs}] Train: {loss_avg.val():0.3f}, Valid: {valid_loss.item():0.3f}'
            loss_avg.reset()
            print(loss_log)
            treatment_model.train()
    return treatment_model


class Treatment(nn.Module):
    '''
    Adds sampling functionality to a Keras model and extends the losses to support
    mixture of gaussian losses.

    # Argument
    '''

    def __init__(self, input_channel, hiddens,dropout,n_components,use_image=False,image_shape=None):
        super(Treatment, self).__init__()
        self.image_shape=image_shape
        self.use_image=use_image
        if use_image:
            self.net=ConvNet(hiddens,dropout=dropout)
        else:
            self.fnn=FeedForward(input_channel,hiddens,dropout)
        self.gaussian=MixtureGaussian(hiddens[-1],n_components)

    def forward(self,inst,feat):
        if self.use_image:
            time, image = feat[:, 0:1].float(), feat[:, 1:].float()
            image=image.reshape(self.image_shape)
            fnn = self.net(image, [time, inst])
        else:
            inp=torch.cat((inst,feat),-1).float()
            fnn=self.fnn(inp)
        out=self.gaussian(fnn)
        return out

    def sample(self,inst,feat,n_samples):
        if self.use_image:
            time, image = feat[:, 0:1].float(), feat[:, 1:].float()
            image=image.reshape(self.image_shape)
            fnn = self.net(image, [time, inst])
        else:
            inp=torch.cat((inst,feat),-1).float()
            fnn=self.fnn(inp)
        out=self.gaussian(fnn)
        # inputs = [i.repeat(n_samples, axis=0) for i in inputs]

        [pi, mu, log_sig] = out
        samples = random_gmm(pi, mu, torch.exp(log_sig))
        return samples


def train_response_kfold(response_model,treatments,y_models,train_datasets,valid_data,dml,batch_size,epochs,lr,w_decay,n_samples=1):

    # train_data = ResponseDataset(z,x,t,y)
    train_loaders=[]
    for data in train_datasets:
        train_loaders.append(torch.utils.data.DataLoader(
        data, batch_size=batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=0,
        pin_memory=True))


    # data parallel for multi-GPU
    # model = torch.nn.DataParallel(model,device_ids=[0,1,2]).to(device)
    response_model = response_model.to(device)
    response_model.train()
    print("Model:")
    print(response_model)


    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, response_model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]
    # epsilon=torch.zeros(1).to(device)
    # epsilon.requires_grad_()
    # filtered_parameters.append(epsilon)
    # setup optimizer
    optimizer = optim.AdamW(filtered_parameters, lr=lr,weight_decay=w_decay)
    # optimizer = optim.SGD(filtered_parameters, lr=0.01,weight_decay=w_decay)

    print("Optimizer:")
    print(optimizer)
    loss_avg = Averager()
    custom_avg=Averager()
    reg_avg=Averager()

    start_time = time.time()
    for ep in range(epochs):
        # train part
        for fold,train_loader in enumerate(train_loaders):
            for i, (inst, feat, treat,resp) in enumerate(train_loader):
                inst = inst.to(device)
                feat = feat.to(device)
                treat = treat.to(device)
                true_res = resp.to(device).float()
                sum_expect=None
                for s in range(n_samples):
                    with torch.no_grad():
                        samples=treatments[fold].sample(inst,feat,n_samples=1)
                        pred_res=y_models[fold](inst,feat)


                # print(samples.shape,feat.shape,treat.shape)
                    if sum_expect is None:
                        sum_expect=response_model(feat,samples)
                    else:
                        sum_expect+= response_model(feat, samples)

                expect=sum_expect/n_samples
                out=response_model(feat,treat)

                if dml==True:
                    square_loss = torch.mean(torch.square(pred_res-expect))
                elif dml==False:
                    square_loss = torch.mean(torch.square(true_res-expect))

                custom_loss1 = torch.mean((out-2*pred_res)*expect)
                # custom_loss2 = torch.mean((expect-2*true_res)*expect)

                # reg_loss=torch.mean(torch.square(true_res-expect-epsilon*(2*true_res-out)))
                # debias_loss=torch.mean(torch.square(true_res-expect)+(out-2*true_res)*(true_res-expect))
                debias_loss=torch.mean(torch.abs(expect*true_res-true_res*true_res))

                # cost=torch.abs(debias_loss)
                cost=square_loss
                response_model.zero_grad()
                cost.backward()
                torch.nn.utils.clip_grad_norm_(response_model.parameters(), 5)  # gradient clipping with 5 (Default)
                optimizer.step()

                loss_avg.add(square_loss)
                custom_avg.add(custom_loss1)
                reg_avg.add(debias_loss)

        if (ep + 1) % 100 == 0 or ep==0:  # To see training progress, we also conduct validation when 'iteration == 0'
            elapsed_time = time.time() - start_time
            # training loss and validation loss
            loss_log = f'[{ep + 1}/{epochs}] Train: {loss_avg.val():0.3f}, Custom:{custom_avg.val():0.3f},Reg:{reg_avg.val():0.3f}, Elapsed_time: {elapsed_time:0.1f}'
            loss_avg.reset()
            custom_avg.reset()
            reg_avg.reset()
            print(loss_log)
    return response_model

def train_response(response_model,treatment_model,Y_model,train_data,valid_data,dml,batch_size,epochs,lr,w_decay,n_samples=1):

    # train_data = ResponseDataset(z,x,t,y)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=0,
        pin_memory=True)

    # data parallel for multi-GPU
    # model = torch.nn.DataParallel(model,device_ids=[0,1,2]).to(device)
    response_model = response_model.to(device)
    response_model.train()
    print("Model:")
    print(response_model)


    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, response_model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]
    epsilon=torch.zeros(1).to(device)
    epsilon.requires_grad_()
    filtered_parameters.append(epsilon)
    # setup optimizer
    optimizer = optim.AdamW(filtered_parameters, lr=lr,weight_decay=w_decay)
    # optimizer = optim.SGD(filtered_parameters, lr=0.01,weight_decay=w_decay)

    print("Optimizer:")
    print(optimizer)
    loss_avg = Averager()
    custom_avg=Averager()
    reg_avg=Averager()

    start_time = time.time()
    for ep in range(epochs):
        # train part
        for i, (inst, feat, treat,resp) in enumerate(train_loader):
            inst = inst.to(device)
            feat = feat.to(device)
            treat = treat.to(device)
            true_res = resp.to(device).float()
            sum_expect=None
            for s in range(n_samples):
                with torch.no_grad():
                    samples=treatment_model.sample(inst,feat,n_samples=1)
                    pred_res=Y_model(inst,feat)


            # print(samples.shape,feat.shape,treat.shape)
                if sum_expect is None:
                    sum_expect=response_model(feat,samples)
                else:
                    sum_expect+= response_model(feat, samples)

            expect=sum_expect/n_samples
            out=response_model(feat,treat)

            if dml==True:
                loss = torch.mean(torch.square(pred_res-expect))
            elif dml==False:
                loss = torch.mean(torch.square(true_res-expect))

            custom_loss1 = torch.mean((out-2*pred_res)*expect)
            # custom_loss2 = torch.mean((expect-2*true_res)*expect)

            reg_loss=torch.mean(torch.square(true_res-expect-epsilon*(2*true_res-out)))
            # debias_loss=torch.mean(torch.square(true_res-expect)+(out-2*true_res)*(true_res-expect))
            debias_loss=torch.mean(torch.abs(expect*true_res-true_res*true_res))

            # cost=torch.abs(debias_loss)
            cost=loss
            response_model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(response_model.parameters(), 5)  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(loss)
            custom_avg.add(custom_loss1)
            reg_avg.add(debias_loss)

        if (ep + 1) % 100 == 0 or ep==0:  # To see training progress, we also conduct validation when 'iteration == 0'
            elapsed_time = time.time() - start_time
            # training loss and validation loss
            loss_log = f'[{ep + 1}/{epochs}] Train: {loss_avg.val():0.3f}, Custom:{custom_avg.val():0.3f},Reg:{reg_avg.val():0.3f}, Elapsed_time: {elapsed_time:0.1f}'
            loss_avg.reset()
            custom_avg.reset()
            reg_avg.reset()
            print(loss_log)
    return response_model

class Response(nn.Module):
    '''
    Adds sampling functionality to a Keras model and extends the losses to support
    mixture of gaussian losses.

    # Argument
    '''

    def __init__(self, input_channel, hiddens, dropout,use_image=False,image_shape=None):
        super(Response, self).__init__()
        self.image_shape=image_shape
        self.use_image=use_image
        if use_image:
            self.net=ConvNet(hiddens,dropout=dropout)
        else:
            self.fnn = FeedForward(input_channel, hiddens, dropout)
        self.output=nn.Linear(hiddens[-1],1)

    def forward(self, feat,treat):
        if self.use_image:
            time,image=feat[:, 0:1], feat[:, 1:]
            image=image.reshape(self.image_shape)
            fnn = self.net(image,[time,treat])
        else:
            inp = torch.cat((feat, treat), -1).float()
            fnn = self.fnn(inp)
        out=self.output(fnn)
        return out
