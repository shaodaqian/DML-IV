from __future__ import print_function
import os
import sys
import time
import random

import warnings
from dmliv.models import Response,Treatment,train_treatment,train_response,Ymodel,train_Ymodel,train_response_kfold
import torch
import torch.optim as optim
from torch.utils.data import random_split,ConcatDataset
import numpy as np
from dmliv.utils import ResponseDataset,device,Averager
# from experiments import data_generator
from data_generator import Gen_fn_IVCluster

import datetime
import pandas as pd


if __name__=='__main__':


    # n = 10000
    epochs = 800

    dropout_rate = 0.1
    # dropout_rate = 0.1

    w_decay = 0.0001
    repeats = 1
    lr = 0.0005
    standard_oos_perf = []
    dml_oos_perf = []
    dml_kf_oos_perf = []
    standard_rewards = []
    dml_rewards = []
    dml_kf_rewards = []
    standard_rewards_ood = []
    dml_rewards_ood = []
    dml_kf_rewards_ood = []
    k_fold = 10
    batch_size = 50

    # config = {
    #     'data': 'fn_IVCluster',
    #     'reps': 10,
    #     'seed': 2022,
    #     'fn': '3dpoly',
    #     'num': 3000,
    #     'numDomain': 5,
    #     'x_dim': 6,
    #     'u_coef': 2,
    #     'x_fn': 'IHDP',
    #     'y_fn': 'n',
    #     'x4u': 0.1,
    #     'dataDir': './real_data/fn_IVCluster/2dpoly/3000_3_3_2_linear_n_0.1/',
    # }
    Gen = Gen_fn_IVCluster()
    Gen.dataDir='./real_data/fn_IVCluster/2dpoly/3000_5_6_2_IHDP_n_0.1/'
    Gen.fn='2dpoly'


    # print(asdf)

    def get_reward(response,g_true,test_data,action_l,action_u,ood=False):
        sample_rate=2000
        samples=np.linspace(action_l, action_u, num=sample_rate)
        samples=torch.tensor(samples).to(device).unsqueeze(-1)
        reward=0
        for x in test_data.x:
            x=np.expand_dims(x, axis=0)
            x_batch=torch.tensor(x).to(device).repeat(sample_rate,1)
            predict=response(x_batch,samples)
            id=torch.argmax(predict).item()
            # print(x.shape,samples[id].shape)
            # print(samples[id])
            reward+=g_true(x,samples[id].cpu().numpy())

        reward=reward/len(test_data.x)
        return reward[0][0]

    def get_error(response,g_true,test_data):
        x, z, t, y=test_data.x,test_data.z,test_data.t,test_data.y
        # t = np.linspace(np.percentile(t, 2.5), np.percentile(t, 97.5), len(x)).reshape(-1, 1)

        # y = g_true(x, z, t)
        y_true = test_data.v
        y_hat = response(x, t).cpu().numpy()
        error=((y_hat - y_true) ** 2).mean()
        print(error)
        # error=((y_hat-g_true(x,t))**2).mean()
        return error

    for i in range(repeats):
        data = Gen.get_exp(i)
        now = datetime.datetime.now()
        date_string = now.strftime("%H%M%S")
        seed = int(date_string)

        torch.manual_seed(seed)
        np.random.seed(seed)


        x, z, t, y = data.train.x,data.train.z,data.train.t,data.train.y
        valid_data= (data.valid.x,data.valid.z,data.valid.t,data.valid.y)
        print(x.max(),x.min())
        print(z.max(),z.min())
        print(t.max(),t.min())
        print(y.max(),y.min())

        test_data=data.valid

        train_dataset=ResponseDataset((z, x, t, y))

        # full_reward=get_reward(lambda x, t: g_true(x,None,t), g_true, test_data, -5, 5)
        # print(full_reward)
        # full_reward=get_reward(lambda x, t: g_true(x,None,t), g_true, ood_test_data, -5, 5)
        # print(full_reward)

        print("Data shapes:\n\
        Features:{x},\n\
        Instruments:{z},\n\
        Treament:{t},\n\
        Response:{y}".format(**{'x':x.shape, 'z':z.shape,
                                't':t.shape, 'y':y.shape}))

        n_components = 5
        loss='mixture_of_gaussians'

        hidden = [128, 64, 32]
        treat_dim=z.shape[-1]+x.shape[-1]


        treatment_model=Treatment(treat_dim,hidden,dropout_rate,n_components)
        treatment_model=train_treatment(treatment_model,train_dataset,valid_data,batch_size,loss,epochs,lr,w_decay)
        treatment_model.eval()

        # epochs = 500
        Y_model=Ymodel(treat_dim,hidden,dropout_rate)
        Y_model=train_Ymodel(Y_model,train_dataset,valid_data,batch_size,epochs,lr,w_decay)
        Y_model.eval()


        # x, z, t, y, g_true = datafunction(int(n), 2)
        # epochs=500
        # batch_size=100
        n_samples=1
        resp_dim=x.shape[-1]+t.shape[-1]
        dml=False
        standard_response=Response(resp_dim,hidden,dropout_rate)
        standard_response=train_response(standard_response,treatment_model,Y_model,train_dataset,valid_data,dml,batch_size,epochs,lr,w_decay,n_samples)
        standard_response.eval()


        dml=True
        dml1_response=Response(resp_dim,hidden,dropout_rate)
        dml1_response=train_response(dml1_response,treatment_model,Y_model,train_dataset,valid_data,dml,batch_size,epochs,lr,w_decay,n_samples)
        dml1_response.eval()



        datasets=random_split(train_dataset, [1/k_fold]*k_fold)
        treatments=[]
        y_models=[]
        for fold in range(k_fold):
            train_data = []
            for d in range(k_fold):
                if d!=fold:
                    train_data.append(datasets[d])

            train_data=ConcatDataset(train_data)
            print(len(train_data))
            treatment_model = Treatment(treat_dim, hidden, dropout_rate, n_components)
            treatment_model = train_treatment(treatment_model, train_data, valid_data, batch_size, loss, epochs,lr, w_decay)
            treatment_model.eval()

            # epochs = 500
            Y_model = Ymodel(treat_dim, hidden, dropout_rate)
            Y_model = train_Ymodel(Y_model, train_data, valid_data, batch_size, epochs,lr, w_decay)
            Y_model.eval()

            treatments.append(treatment_model)
            y_models.append(Y_model)

        dml=True
        dml_k_response=Response(resp_dim,hidden,dropout_rate)
        dml_k_response=train_response_kfold(dml_k_response,treatments,y_models,datasets,valid_data,dml,batch_size,epochs,lr,w_decay,n_samples)
        dml_k_response.eval()


        with torch.no_grad():
            standard_mse=get_error(lambda x,t: standard_response(torch.tensor(x).to(device),torch.tensor(t).to(device)),Gen.ihdp_g,test_data)
            dml_mse=get_error(lambda x,t: dml1_response(torch.tensor(x).to(device),torch.tensor(t).to(device)),Gen.ihdp_g,test_data)
            dml_k_mse=get_error(lambda x,t: dml_k_response(torch.tensor(x).to(device),torch.tensor(t).to(device)),Gen.ihdp_g,test_data)
            # print("Standard oos: %f" % standard_mse)
            print("CE-DML-IV oos: %f" % dml_mse)
            print("DML-IV oos: %f" % dml_k_mse)

            standard_oos_perf.append(standard_mse)
            dml_oos_perf.append(dml_mse)
            dml_kf_oos_perf.append(dml_k_mse)

            standard_r=get_reward(standard_response,Gen.ihdp_g,test_data,-3,5)
            dml_r=get_reward(dml1_response,Gen.ihdp_g,test_data,-3,5)
            dml_kf_r=get_reward(dml_k_response,Gen.ihdp_g,test_data,-3,5)
            # print('standard reward: ',standard_r)
            print('CE-DML-IV reward: ',dml_r)
            print('DML-IV reward: ',dml_kf_r)


            standard_rewards.append(standard_r)
            dml_rewards.append(dml_r)
            dml_kf_rewards.append(dml_kf_r)



    all_results=[standard_oos_perf,dml_oos_perf,dml_kf_oos_perf,
                 standard_rewards,dml_rewards,dml_kf_rewards]
    all_results=np.array(all_results)
