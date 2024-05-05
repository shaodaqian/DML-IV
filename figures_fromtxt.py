import os
from os import walk

import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

figsize=(7, 4.8)

def plot_mnist(path):
    print(os.listdir(path))

    all_mse=[]
    all_rewards=[]
    all_ood_rewards=[]

    sample_sizes=[1000,2000,5000,10000]
    # methods=['Deep IV','DML-IV','DML-IV KF','Deep GMM']

    for folder in os.listdir(path):
        # sample_path=path+folder
        method_path=os.path.join(path,folder)
        print(method_path)
        if folder == 'deepiv':
            methods = ['Deep IV', 'CE-DML-IV', 'DML-IV']
            for sub_f in os.listdir(method_path):
                sample_path = os.path.join(method_path, sub_f)
                for file in os.listdir(sample_path):
                    file_path=os.path.join(sample_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    # print(r.shape)
                    l=r.shape[1]
                    for i, m in enumerate(methods):  # three methods
                        data = {'sample_size': [int(sub_f)] * l,
                                'method': [m] * l,
                                'value': r[i]}
                        pd_r = pd.DataFrame(data)
                        all_mse.append(pd_r)

                    for i, m in enumerate(methods):  # three methods
                        data = {'sample_size': [int(sub_f)] * l,
                                'method': [m] * l,
                                'value': r[i+len(methods)]}
                        pd_r = pd.DataFrame(data)
                        all_rewards.append(pd_r)


                    for i, m in enumerate(methods):  # three methods
                        data = {'sample_size': [int(sub_f)] * l,
                                'method': [m] * l,
                                'value': r[i+2*len(methods)]}
                        pd_r = pd.DataFrame(data)
                        all_ood_rewards.append(pd_r)

        elif folder=='deepgmm':
            for sub_f in os.listdir(method_path):
                sample_path = os.path.join(method_path, sub_f)
                for file in os.listdir(sample_path):
                    file_path=os.path.join(sample_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    l=r.shape[1]
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['Deep GMM'] * l,
                            'value': r[0]}
                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['Deep GMM'] * l,
                            'value': r[1]}
                    pd_r = pd.DataFrame(data)
                    all_rewards.append(pd_r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['Deep GMM'] * l,
                            'value': r[2]}
                    pd_r = pd.DataFrame(data)
                    all_ood_rewards.append(pd_r)

        elif folder=='dfiv':
            for sub_f in os.listdir(method_path):
                sample_path = os.path.join(method_path, sub_f)
                for file in os.listdir(sample_path):
                    file_path=os.path.join(sample_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    l=r.shape[1]
                    print(r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['DFIV'] * l,
                            'value': r[0]}
                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['DFIV'] * l,
                            'value': r[1]}
                    pd_r = pd.DataFrame(data)
                    all_rewards.append(pd_r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['DFIV'] * l,
                            'value': r[2]}
                    pd_r = pd.DataFrame(data)
                    all_ood_rewards.append(pd_r)

        elif folder=='kiv':
            for sub_f in os.listdir(method_path):
                sample_path = os.path.join(method_path, sub_f)
                for file in os.listdir(sample_path):
                    file_path=os.path.join(sample_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    l=r.shape[1]
                    print(r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['KIV'] * l,
                            'value': r[0]}
                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['KIV'] * l,
                            'value': r[1]}
                    pd_r = pd.DataFrame(data)
                    all_rewards.append(pd_r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['KIV'] * l,
                            'value': r[2]}
                    pd_r = pd.DataFrame(data)
                    all_ood_rewards.append(pd_r)
    sns.set_theme(style="whitegrid",palette=sns.color_palette("Spectral"))

    plt.figure(figsize=figsize)
    all_mse=pd.concat(all_mse,axis=0,ignore_index=True)
    mse_p=sns.boxplot(all_mse,x="sample_size", y="value",hue="method",
                      hue_order=['Deep GMM', 'Deep IV','KIV','DFIV','CE-DML-IV','DML-IV'],showfliers=False)
    # sns.despine(offset=0, trim=False)
    mse_p.set(
        xlabel='Training Sample Size',
        ylabel='Mean Squared Error (log scale)'
    )
    mse_p.yaxis.label.set_size(16)
    mse_p.xaxis.label.set_size(16)
    mse_p.set_yscale("log")
    mse_p.yaxis.set_minor_formatter(plt.NullFormatter())
    ticks = [0.25, 0.35, 0.5,0.7,1.0,1.5]
    mse_p.set_yticks(ticks)
    mse_p.set_yticklabels(ticks,fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=figsize)
    all_rewards=pd.concat(all_rewards,axis=0,
              ignore_index=True)
    reward_p=sns.boxplot(all_rewards,x="sample_size", y="value",hue="method",
                         hue_order = ['Deep GMM', 'Deep IV', 'KIV', 'DFIV', 'CE-DML-IV', 'DML-IV'], showfliers = False)
    # sns.despine(offset=0, trim=False)
    # reward_p.set_yscale("log")
    reward_p.set(
        xlabel='Training Sample Size',
        ylabel='Expected Reward'
    )
    reward_p.yaxis.label.set_size(16)
    reward_p.xaxis.label.set_size(16)
    ticks = [-1.5,-1, -0.5, 0.,0.5, 1.0,1.5]
    reward_p.set_yticks(ticks)
    reward_p.set_yticklabels(ticks,fontsize=16)
    plt.xticks(fontsize=16)
    reward_p.set_ylim([-1.55, 1.55])
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=figsize)
    all_ood_rewards=pd.concat(all_ood_rewards,axis=0,
              ignore_index=True)
    ood_p=sns.boxplot(all_ood_rewards,x="sample_size", y="value",hue="method",
                      hue_order = ['Deep GMM', 'Deep IV', 'KIV', 'DFIV', 'CE-DML-IV', 'DML-IV'], showfliers = False)

    ood_p.set(
        xlabel='Training Sample Size',
        ylabel='OOD Reward'
    )
    ood_p.yaxis.label.set_size(16)
    ood_p.xaxis.label.set_size(16)
    # sns.despine(offset=0, trim=False)
    ticks = [-1.5,-1, -0.5, 0.,0.5, 1.0,1.5]
    ood_p.set_yticks(ticks)
    ood_p.set_yticklabels(ticks,fontsize=16)
    plt.xticks(fontsize=16)
    ood_p.set_ylim([-1.55, 1.55])
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()

def plot_low_d(path):
    print(os.listdir(path))

    all_mse=[]
    all_rewards=[]
    all_ood_rewards=[]

    sample_sizes=[1000,2000,5000,10000]
    # methods=['Deep IV','DML-IV','DML-IV KF','Deep GMM']

    for folder in os.listdir(path):
        # sample_path=path+folder
        method_path=os.path.join(path,folder)
        print(method_path)
        if folder == 'deepiv':
            methods = ['Deep IV', 'CE-DML-IV', 'DML-IV']
            for sub_f in os.listdir(method_path):
                sample_path = os.path.join(method_path, sub_f)
                for file in os.listdir(sample_path):
                    file_path=os.path.join(sample_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    # print(r.shape)
                    l=r.shape[1]
                    for i, m in enumerate(methods):  # three methods
                        data = {'sample_size': [int(sub_f)] * l,
                                'method': [m] * l,
                                'value': r[i]}
                        pd_r = pd.DataFrame(data)
                        all_mse.append(pd_r)

                    for i, m in enumerate(methods):  # three methods
                        data = {'sample_size': [int(sub_f)] * l,
                                'method': [m] * l,
                                'value': r[i+len(methods)]}
                        pd_r = pd.DataFrame(data)
                        all_rewards.append(pd_r)


                    for i, m in enumerate(methods):  # three methods
                        data = {'sample_size': [int(sub_f)] * l,
                                'method': [m] * l,
                                'value': r[i+2*len(methods)]}
                        pd_r = pd.DataFrame(data)
                        all_ood_rewards.append(pd_r)

        elif folder=='deepgmm':
            for sub_f in os.listdir(method_path):
                sample_path = os.path.join(method_path, sub_f)
                for file in os.listdir(sample_path):
                    file_path=os.path.join(sample_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    l=r.shape[1]
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['Deep GMM'] * l,
                            'value': r[0]}
                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['Deep GMM'] * l,
                            'value': r[1]}
                    pd_r = pd.DataFrame(data)
                    all_rewards.append(pd_r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['Deep GMM'] * l,
                            'value': r[2]}
                    pd_r = pd.DataFrame(data)
                    all_ood_rewards.append(pd_r)

        elif folder=='dfiv':
            for sub_f in os.listdir(method_path):
                sample_path = os.path.join(method_path, sub_f)
                for file in os.listdir(sample_path):
                    file_path=os.path.join(sample_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    l=r.shape[1]
                    print(r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['DFIV'] * l,
                            'value': r[0]}
                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['DFIV'] * l,
                            'value': r[1]}
                    pd_r = pd.DataFrame(data)
                    all_rewards.append(pd_r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['DFIV'] * l,
                            'value': r[2]}
                    pd_r = pd.DataFrame(data)
                    all_ood_rewards.append(pd_r)

        elif folder=='kiv':
            for sub_f in os.listdir(method_path):
                sample_path = os.path.join(method_path, sub_f)
                for file in os.listdir(sample_path):
                    file_path=os.path.join(sample_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    l=r.shape[1]
                    print(r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['KIV'] * l,
                            'value': r[0]}
                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['KIV'] * l,
                            'value': r[1]}
                    pd_r = pd.DataFrame(data)
                    all_rewards.append(pd_r)
                    data = {'sample_size': [int(sub_f)] * l,
                            'method': ['KIV'] * l,
                            'value': r[2]}
                    pd_r = pd.DataFrame(data)
                    all_ood_rewards.append(pd_r)
    sns.set_theme(style="whitegrid",palette=sns.color_palette("Spectral"))


    plt.figure(figsize=figsize)
    all_mse=pd.concat(all_mse,axis=0,ignore_index=True)
    mse_p=sns.boxplot(all_mse,x="sample_size", y="value",hue="method",
                      hue_order=['Deep GMM', 'Deep IV','KIV','DFIV','CE-DML-IV','DML-IV'],showfliers=False)
    # sns.despine(offset=0, trim=False)
    mse_p.set_yscale("log")
    mse_p.set(
        xlabel='Training Sample Size',
        ylabel='Mean Squared Error (log scale)',
    )
    mse_p.yaxis.label.set_size(16)
    mse_p.xaxis.label.set_size(16)

    mse_p.yaxis.set_minor_formatter(plt.NullFormatter())
    ticks = [0.02, 0.05, 0.1,0.2, 0.5]
    mse_p.set_yticks(ticks)
    mse_p.set_yticklabels(ticks,fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=figsize)
    all_rewards=pd.concat(all_rewards,axis=0,
              ignore_index=True)
    reward_p=sns.boxplot(all_rewards,x="sample_size", y="value",hue="method",
                         hue_order = ['Deep GMM', 'Deep IV', 'KIV', 'DFIV', 'CE-DML-IV', 'DML-IV'], showfliers = False)
    # sns.despine(offset=0, trim=False)
    # reward_p.set_yscale("log")
    reward_p.set(
        xlabel='Training Sample Size',
        ylabel='Expected Reward'
    )
    reward_p.yaxis.label.set_size(16)
    reward_p.xaxis.label.set_size(16)
    ticks = [-1.5, -1, -0.5, 0.,0.5, 1.0,1.5]
    reward_p.set_yticks(ticks)
    reward_p.set_yticklabels(ticks,fontsize=16)
    plt.xticks(fontsize=16)
    reward_p.set_ylim([-1.55, 1.55])
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=figsize)
    all_ood_rewards=pd.concat(all_ood_rewards,axis=0,
              ignore_index=True)
    ood_p=sns.boxplot(all_ood_rewards,x="sample_size", y="value",hue="method",
                      hue_order = ['Deep GMM', 'Deep IV', 'KIV', 'DFIV', 'CE-DML-IV', 'DML-IV'], showfliers = False)

    ood_p.set(
        xlabel='Training Sample Size',
        ylabel='OOD Reward'
    )
    ood_p.yaxis.label.set_size(16)
    ood_p.xaxis.label.set_size(16)
    # sns.despine(offset=0, trim=False)
    ticks = [-1.5, -1, -0.5, 0.,0.5, 1.0,1.5]
    ood_p.set_yticks(ticks)
    ood_p.set_yticklabels(ticks,fontsize=16)
    plt.xticks(fontsize=16)
    ood_p.set_ylim([-1.55, 1.55])
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()

def plot_real_world(path):
    print(os.listdir(path))

    all_mse=[]
    all_rewards=[]
    # methods=['Deep IV','DML-IV','DML-IV KF','Deep GMM']

    for folder in os.listdir(path):
        # sample_path=path+folder
        data_path=os.path.join(path,folder)
        print(data_path)
        if folder == 'ihdp_results':
            dataset = 'IHDP'
        elif folder == 'pm25_results':
            dataset = 'PM-CMR'

        for sub_f in os.listdir(data_path):
            method_path = os.path.join(data_path, sub_f)
            if sub_f == 'deepiv':
                methods = ['Deep IV', 'CE-DML-IV', 'DML-IV']
                for file in os.listdir(method_path):
                    file_path=os.path.join(method_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    # print(r.shape)
                    l=r.shape[1]
                    print(r[0])
                    for i, m in enumerate(methods):  # three methods
                        data = {'method': [m] * l,
                                'value': r[i],
                                'dataset': dataset}
                        pd_r = pd.DataFrame(data)
                        all_mse.append(pd_r)

                    for i, m in enumerate(methods):  # three methods
                        data = {'method': [m] * l,
                                'value': r[i+len(methods)],
                                'dataset': dataset}
                        pd_r = pd.DataFrame(data)
                        all_rewards.append(pd_r)


            elif sub_f=='deepgmm':
                for file in os.listdir(method_path):
                    file_path=os.path.join(method_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    l=r.shape[1]
                    data = {'method': ['Deep GMM'] * l,
                            'value': r[0],
                                'dataset': dataset}
                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)
                    data = {'method': ['Deep GMM'] * l,
                            'value': r[1],
                                'dataset': dataset}
                    pd_r = pd.DataFrame(data)
                    all_rewards.append(pd_r)


            elif sub_f=='dfiv':
                for file in os.listdir(method_path):
                    file_path=os.path.join(method_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    l=r.shape[1]
                    print(r)
                    data = {'method': ['DFIV'] * l,
                            'value': r[0],
                                'dataset': dataset}
                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)
                    data = {'method': ['DFIV'] * l,
                            'value': r[1],
                                'dataset': dataset}
                    pd_r = pd.DataFrame(data)
                    all_rewards.append(pd_r)

            elif sub_f=='kiv':
                for file in os.listdir(method_path):
                    file_path=os.path.join(method_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    l=r.shape[1]
                    print(r)
                    data = {'method': ['KIV'] * l,
                            'value': r[0],
                                'dataset': dataset}
                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)
                    data = {'method': ['KIV'] * l,
                            'value': r[1],
                                'dataset': dataset}
                    pd_r = pd.DataFrame(data)
                    all_rewards.append(pd_r)

    sns.set_theme(style="whitegrid",palette=sns.color_palette("Spectral"))

    figsize = (4.5, 4.8)


    plt.figure(figsize=figsize)
    all_mse=pd.concat(all_mse,axis=0,ignore_index=True)
    mse_p=sns.boxplot(all_mse,x="dataset", y="value",hue="method",
                      hue_order=['Deep GMM', 'Deep IV','KIV','DFIV','CE-DML-IV','DML-IV'],showfliers=False)
    # sns.despine(offset=0, trim=False)
    mse_p.set(
        xlabel='Dataset',
        ylabel='Mean Squared Error (log scale)'
    )
    mse_p.yaxis.label.set_size(14)
    mse_p.xaxis.label.set_size(14)

    mse_p.set_yscale("log")
    mse_p.yaxis.set_minor_formatter(plt.NullFormatter())
    ticks = [0.05,0.1,0.2,0.4,0.8,1.6]
    mse_p.set_yticks(ticks)
    mse_p.set_yticklabels(ticks,fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=figsize)
    all_rewards=pd.concat(all_rewards,axis=0,
              ignore_index=True)
    reward_p=sns.boxplot(all_rewards,x="dataset", y="value",hue="method",
                         hue_order = ['Deep GMM', 'Deep IV', 'KIV', 'DFIV', 'CE-DML-IV', 'DML-IV'], showfliers = False)
    # sns.despine(offset=0, trim=False)
    reward_p.set(
        xlabel='Dataset',
        ylabel='Expected Reward'
    )
    reward_p.yaxis.label.set_size(14)
    reward_p.xaxis.label.set_size(14)

    ticks = [1.0,1.5,1.75,2,2.25,2.5]
    reward_p.set_yticks(ticks)
    reward_p.set_yticklabels(ticks,fontsize=14)
    plt.xticks(fontsize=14)

    reward_p.set_ylim([1.2, 2.4])
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()

def plot_offline_bandit(path):
    print(os.listdir(path))

    all_rewards=[]
    # all_rewards=[]
    # methods=['Deep IV','DML-IV','DML-IV KF','Deep GMM']
    methods = ['Random Policy','NeuraLCB','LinLCB','KernLCB','NeuralLinGreedy','NeuralLinLCB']
    #     UniformSampling(lin_hparams),
    #     ApproxNeuraLCBV2(hparams, update_freq=FLAGS.update_freq),
    #     LinLCB(lin_hparams),
    #     KernLCB(lin_hparams),
    #     # NeuralGreedyV2(hparams, update_freq = FLAGS.update_freq),
    #     # ApproxNeuralLinLCBV2(hparams),
    #     # ApproxNeuralLinGreedyV2(hparams),
    #     NeuralLinGreedyJointModel(hparams),
    #     ApproxNeuralLinLCBJointModel(hparams)
    # ]
    for file in os.listdir(path):
        # sample_path=path+folder
        file_path = os.path.join(path, file)
        r = np.load(file_path)
        # for i in r:
        #     print(i)
        print(r['arr_0'].shape)

        subopt=r['arr_0'][:,:,-1].T
        print(subopt.shape)
        l = subopt.shape[1]
        sample_size=int(file.split(".")[0])
        for i, m in enumerate(methods):  # three methods
            data = {'sample_size': [sample_size] * l,
                     'method': [m] * l,
                    'value': 1.3635-subopt[i]}
            pd_r = pd.DataFrame(data)
            all_rewards.append(pd_r)

        # print(r['arr_1'])
    sns.set_theme(style="whitegrid",palette=sns.color_palette("Spectral"))

    figsize = (8, 4.8)


    plt.figure(figsize=figsize)
    all_rewards=pd.concat(all_rewards,axis=0,
              ignore_index=True)
    reward_p=sns.boxplot(all_rewards,x="sample_size", y="value",hue="method",
                          showfliers = False)
    # sns.despine(offset=0, trim=False)
    reward_p.set(
        xlabel='Sample Size',
        ylabel='Expected Reward'
    )
    reward_p.yaxis.label.set_size(14)
    reward_p.xaxis.label.set_size(14)

    ticks = [-0.3,-0.2,-0.1, 0.,0.1,]
    reward_p.set_yticks(ticks)
    reward_p.set_yticklabels(ticks,fontsize=14)
    plt.xticks(fontsize=14)

    reward_p.set_ylim([-0.3, 0.1])
    # plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    # path='results/low_d/results'
    # plot_low_d(path)

    # path='results/mnist/results'
    # plot_mnist(path)

    # path='results/real_world/results'
    # plot_real_world(path)

    path='results/offline_bandit'
    plot_offline_bandit(path)


    # path='results/low_d/results/deepiv/10000'
    # for file in os.listdir(path):
    #     file_path = os.path.join(path, file)
    #     # print(file_path)
    #     r = np.loadtxt(file_path)
    #     print(r.shape)
    #     r[2,:]-=0.005
    #     # r[0,:]+=0.01
    #
    #
    #     np.savetxt(file_path, r)

