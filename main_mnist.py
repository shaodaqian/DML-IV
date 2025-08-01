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
import data_generator
import datetime


if __name__=='__main__':


    n = 10000
    # epochs = int(500000./float(n))+400 # heuristic to select number of epochs
    epochs=500
    use_image=True
    dropout_rate = 0.3
    # dropout_rate = min(1000. / (1000. + n), 0.5)

    w_decay = 0.05
    repeats = 10
    lr = 0.001
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
    batch_size = 128
    image_shape = (-1,1, 28, 28)
    # test_seed=np.random.randint(1e6)
    test_seed=66666


    def datafunction(n, s, images=use_image,test=False, ood=False):
        return data_generator.deep_IV_data(n=n, seed=s, ypcor=0.9, use_images=images, ood=ood, test=test)


    def get_reward(response,g_true,test_seed,action_l,action_u,images,ood=False):
        test_data = datafunction(5000, test_seed,images=images,ood=ood,test=True)

        if images:
            x_latent, _, _, _, _ = datafunction(5000, test_seed, images=False,ood=ood,test=True)

        sample_rate=2000
        batch_size=1000
        samples=np.linspace(action_l, action_u, num=sample_rate)
        samples=torch.tensor(samples).to(device).unsqueeze(-1)
        reward=0
        for idx,x in enumerate(test_data[0]):
            x=np.expand_dims(x, axis=0)
            x_batch=torch.tensor(x).to(device).repeat(sample_rate,1)
            # predict=[]
            # for i in range(int(sample_rate/batch_size)):
            #     pred=response(x_batch[i*batch_size:(i+1)*batch_size],samples[i*batch_size:(i+1)*batch_size])
            #     predict.append(pred)
            # predict=torch.concat(predict,dim=0)
            predict=response(x_batch,samples)
            # print(predict.shape)
            id=torch.argmax(predict).item()
            # print(x.shape,samples[id].shape)
            # print(samples[id])
            # print(x.shape)
            # print(x_latent[idx:idx+1,:].shape)
            reward+=g_true(x_latent[idx:idx+1,:],None,samples[id].cpu().numpy())

        reward=reward/len(test_data[0])
        return reward[0][0]

    for i in range(repeats):

        now = datetime.datetime.now()
        date_string = now.strftime("%H%M%S")
        seed = int(date_string)

        torch.manual_seed(seed)
        np.random.seed(seed)

        x, z, t, y, g_true = datafunction(int(n), seed)
        valid_data = datafunction(1000, 1111)

        train_data=z,x,t,y
        train_dataset=ResponseDataset(train_data)


        print("Data shapes:\n\
        Features:{x},\n\
        Instruments:{z},\n\
        Treament:{t},\n\
        Response:{y}".format(**{'x':x.shape, 'z':z.shape,
                                't':t.shape, 'y':y.shape}))

        n_components = 10
        loss='mixture_of_gaussians'

        hidden = [128, 64, 32]
        treat_dim=z.shape[-1]+x.shape[-1]


        treatment_model=Treatment(treat_dim,hidden,dropout_rate,n_components,use_image,image_shape)
        treatment_model=train_treatment(treatment_model,train_dataset,valid_data,batch_size,loss,epochs,lr,w_decay)
        treatment_model.eval()

        # epochs = 500
        Y_model=Ymodel(treat_dim,hidden,dropout_rate,use_image,image_shape)
        Y_model=train_Ymodel(Y_model,train_dataset,valid_data,batch_size,epochs,lr,w_decay)
        Y_model.eval()


        # x, z, t, y, g_true = datafunction(int(n), 2)
        # epochs=500
        # batch_size=100
        n_samples=1
        resp_dim=x.shape[-1]+t.shape[-1]
        dml=False
        standard_response=Response(resp_dim,hidden,dropout_rate,use_image,image_shape)
        standard_response=train_response(standard_response,treatment_model,Y_model,train_dataset,valid_data,dml,batch_size,epochs,lr,w_decay,n_samples)
        standard_response.eval()


        dml=True
        dml1_response=Response(resp_dim,hidden,dropout_rate,use_image,image_shape)
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
            # print(len(train_data))
            treatment_model = Treatment(treat_dim, hidden, dropout_rate, n_components,use_image,image_shape)
            treatment_model = train_treatment(treatment_model, train_data, valid_data, batch_size, loss, epochs,lr, w_decay)
            treatment_model.eval()

            # epochs = 500
            Y_model = Ymodel(treat_dim, hidden, dropout_rate,use_image,image_shape)
            Y_model = train_Ymodel(Y_model, train_data, valid_data, batch_size, epochs,lr, w_decay)
            Y_model.eval()

            treatments.append(treatment_model)
            y_models.append(Y_model)

        dml=True
        dml_k_response=Response(resp_dim,hidden,dropout_rate,use_image,image_shape)
        dml_k_response=train_response_kfold(dml_k_response,treatments,y_models,datasets,valid_data,dml,batch_size,epochs,lr,w_decay,n_samples)
        dml_k_response.eval()


        with torch.no_grad():
            standard_mse = data_generator.get_error(lambda x,t: standard_response(torch.tensor(x).to(device),torch.tensor(t).to(device)),g_true,
                                                    image=images, ood=False)
            dml_mse = data_generator.get_error(lambda x,t: dml1_response(torch.tensor(x).to(device),torch.tensor(t).to(device)),g_true,
                                                    image=images, ood=False)
            dml_k_mse = data_generator.get_error(lambda x,t: dml_k_response(torch.tensor(x).to(device),torch.tensor(t).to(device)),g_true,
                                                    image=images, ood=False)

            oos_perf = data_generator.monte_carlo_error([lambda x,z,t: standard_response(torch.tensor(x).to(device),torch.tensor(t).to(device)),
                                                         lambda x,z,t: dml1_response(torch.tensor(x).to(device),torch.tensor(t).to(device)),
                                                         # lambda x, z, t: dml2_response(torch.tensor(x).to(device),torch.tensor(t).to(device)),
                                                         lambda x,z,t: dml_k_response(torch.tensor(x).to(device),torch.tensor(t).to(device))
                                                         ],
                                                    datafunction, has_latent=use_image, ood=False)
            # print("Standard oos: %f" % oos_perf[0])
            print("CE-DML-IV oos: %f" % oos_perf[1])
            print("DML-IV oos: %f" % oos_perf[2])

            standard_oos_perf.append(oos_perf[0])
            dml_oos_perf.append(oos_perf[1])
            dml_kf_oos_perf.append(oos_perf[2])

            standard_r=get_reward(standard_response,g_true,test_seed,-3,3,use_image)
            dml_r=get_reward(dml1_response,g_true,test_seed,-3,3,use_image)
            dml_kf_r=get_reward(dml_k_response,g_true,test_seed,-3,3,use_image)
            # print('standard reward: ',standard_r)
            print('CE-DML-IV reward: ',dml_r)
            print('DML-IV reward: ',dml_kf_r)

            standard_ood_r=get_reward(standard_response,g_true,test_seed,-3,3,use_image,ood=True)
            dml_ood_r=get_reward(dml1_response,g_true,test_seed,-3,3,use_image,ood=True)
            dml_kf_ood_r=get_reward(dml_k_response,g_true,test_seed,-3,3,use_image,ood=True)
            # print('standard reward: ',standard_ood_r)
            print('CE-DML-IV reward: ',dml_ood_r)
            print('DML-IV reward: ',dml_kf_ood_r)

            standard_rewards.append(standard_r)
            dml_rewards.append(dml_r)
            dml_kf_rewards.append(dml_kf_r)

            standard_rewards_ood.append(standard_ood_r)
            dml_rewards_ood.append(dml_ood_r)
            dml_kf_rewards_ood.append(dml_kf_ood_r)

    # print(sdf)

    all_results=[standard_oos_perf,dml_oos_perf,dml_kf_oos_perf,
                 standard_rewards,dml_rewards,dml_kf_rewards,
                 standard_rewards_ood,dml_rewards_ood,dml_kf_rewards_ood]
    all_results=np.array(all_results)

    # print(sdf)
    foldername = str(datetime.datetime.now().strftime("%m-%d"))
    dump_dir = f'./results/mnist/{foldername}/{n}'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
        
    np.savetxt(f'{dump_dir}/result_{seed}.txt', all_results)
    b = np.loadtxt(f'{dump_dir}/result_{seed}.txt')
    print(b)
