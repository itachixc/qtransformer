import argparse
import os
import re
from itertools import groupby
import scienceplots

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mmpretrain.utils import load_json_log


def get_log_dicts(filename):
    json_logs = filename
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = [load_json_log(json_log) for json_log in json_logs]
    return log_dicts




# def plot_curve(log_dicts, args):
#     """Plot train metric-iter graph."""
#     # set style
#     try:
#         import seaborn as sns
#         sns.set_style(args.style)
#     except ImportError:
#         pass

#     # set plot window size
#     wind_w, wind_h = args.window_size.split('*')
#     wind_w, wind_h = int(wind_w), int(wind_h)
#     plt.figure(figsize=(wind_w, wind_h))

#     # get legends and metrics
#     legends = 'accu'
#     metrics = args.keys

#     # plot curves from log_dicts by metrics
#     plot_curve_helper(log_dicts, metrics, args, legends)

#     # set title and show or save
#     if args.title is not None:
#         plt.title(args.title)
#     if args.out is None:
#         plt.show()
#     else:
#         print(f'save curve to: {args.out}')
#         plt.savefig(args.out)
#         plt.cla()

def plot_loss(log_dicts):
    log_train=log_dicts['train']
    loss=[log['loss'] for log in log_train]
    iterations=[log['step'] for log in log_train]
    plt.plot(iterations,loss)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()

def plot_accu(log_dicts):
    log_train=log_dicts['val']
    accu=[log['accuracy/top1'] for log in log_train]
    iterations=[50*i for i,_ in enumerate(log_train)]
    plt.plot(iterations,accu)
    plt.xlabel('iterations')
    plt.ylabel('accu')
    plt.show()

def get_eps(filename):
    eps=[]
    for fname in filename:
        temp1=fname.index('eps')+3
        temp2=fname.index('/',temp1,temp1+10)
        eps.append(float(fname[temp1:temp2]))
    return eps


def plot_loss_curve(filename,dataset='cifar10',kind='train'):
    log_dicts=get_log_dicts(filename)
    eps=get_eps(filename)
    if kind=='train':
        y_label=['loss','loss']
    elif kind=='val':
        y_label=['accu','accuracy/top1']
    for i in range(len(log_dicts)):
        logs=log_dicts[i][kind]
        if kind=='train':
            iterations=[log['step'] for log in logs]
        else:
            iterations=[50*i for i,_ in enumerate(logs)]
        y=[log[y_label[1]] for log in logs]
        # sns.lineplot(iterations,y,label='eps={}'.format(eps[i]))
        plt.plot(iterations,y,label='eps={}'.format(eps[i]))
    plt.xlabel('iterations')
    plt.ylabel(y_label[1])
    plt.legend(loc=1)
    plt.savefig(f'results/{dataset}/eps_to_{y_label[0]}.png')
    # plt.show()
def plot_curve_defer(filename,filename_defer,dataset,kind='train'):
    log_dicts=get_log_dicts(filename)
    log_dicts_defer=get_log_dicts(filename_defer)
    eps=get_eps(filename)
    color_list=['b','g','r','c','m','y','k','w']
    plt.style.use(['science','ieee'])
    # plt.style.use('science')
    s_=2.5
    if kind=='train':
        y_label=['loss','loss']
    elif kind=='val':
        y_label=['accu','accuracy/top1']
    for i in range(len(log_dicts)):
        logs=log_dicts[i][kind]
        
        if kind=='train':
            iterations=[log['step'] for log in logs]
        else:
            iterations=[20*i for i,_ in enumerate(logs)]
        y=[log[y_label[1]] for log in logs]
        # sns.lineplot(iterations,y,label='eps={}'.format(eps[i]))
        plt.plot(iterations,y,label='$\delta$={}'.format(eps[i]),color=color_list[i],linewidth=s_)
      
    for i in range(len(log_dicts_defer)):
        
        logs_defer=log_dicts_defer[i][kind]
        if kind=='train':
            iterations=[log['step'] for log in logs_defer]
        else:
            iterations=[20*i for i,_ in enumerate(logs_defer)]
        y_defer=[log[y_label[1]] for log in logs_defer]
        # sns.lineplot(iterations,y,label='eps={}'.format(eps[i]))
        plt.plot(iterations,y_defer,linestyle=':',color=color_list[i],linewidth=s_)
        # plt.plot(iterations,y_defer,label='defer eps={}'.format(eps[i]),linestyle=':',color=color_list[i])
    plt.tick_params(axis='both',labelsize=10)
    plt.xlabel('iterations',size=20)
    plt.ylabel(y_label[1],size=20)
    # plt.title(dataset)
    plt.legend(loc=1)
    plt.savefig(f'results/{dataset}/{dataset}_compare_eps_to_{y_label[0]}.png')
    plt.close()



if __name__=="__main__":
    # filename=['work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps0/20231220_104047/vis_data/20231220_104047.json']
    filename_cifar10=[
            #   'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps1e-2/20231221_001721/vis_data/20231221_001721.json',
              'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps5e-3/20231221_001731/vis_data/20231221_001731.json',
              'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps3e-3/20231221_091823/vis_data/20231221_091823.json',
            #   'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps25e-4/20231221_160556/vis_data/20231221_160556.json',
              'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps2e-3/20231221_091725/vis_data/20231221_091725.json',
              'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps1e-3/20231220_143917/vis_data/20231220_143917.json',
              'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps1e-4/20231220_143935/vis_data/20231220_143935.json',
              'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps0/20231220_104047/vis_data/20231220_104047.json',
              ]
    filename_cifar10_defer=[
            #   'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps1e-2_defer/20231224_094114/vis_data/20231224_094114.json',
              'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps5e-3_defer/20231223_082410/vis_data/20231223_082410.json',
              'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps3e-3_defer/20231228_221811/vis_data/20231228_221811.json',
            #   'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps25e-4_defer/20231222_161738/vis_data/20231222_161738.json',
              'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps2e-3_defer/20231222_091434/vis_data/20231222_091434.json',
              'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps1e-3_defer/20231222_091625/vis_data/20231222_091625.json',
              'work_dirs_20240104/vit_config_cifar10_bs64_p16_384_eps1e-4_defer/20231223_082205/vis_data/20231223_082205.json',
              
              ]
    filename_cifar100=[
            #   'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps1e-2/20231228_185107/vis_data/20231228_185107.json',
              'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps5e-3/20231227_185319/vis_data/20231227_185319.json',
              'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps3e-3/20231225_213107/vis_data/20231225_213107.json',
            #   'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps25e-4/20231226_092031/vis_data/20231226_092031.json',
              'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps2e-3/20231225_213039/vis_data/20231225_213039.json',
              'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps1e-3/20231225_090815/vis_data/20231225_090815.json',
              'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps1e-4/20231225_090757/vis_data/20231225_090757.json',
              'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps0/20231224_094538/vis_data/20231224_094538.json',
              ]
    filename_cifar100_defer=[
            #   'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps1e-2_defer/20231228_230423/vis_data/20231228_230423.json',
              'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps5e-3_defer/20231228_092519/vis_data/20231228_092519.json',
              'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps3e-3_defer/20231227_091425/vis_data/20231227_091425.json',
            #   'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps25e-4_defer/20231227_105922/vis_data/20231227_105922.json',
              'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps2e-3_defer/20231226_213722/vis_data/20231226_213722.json',
              'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps1e-3_defer/20231226_213651/vis_data/20231226_213651.json',
              'work_dirs_20240104/vit_config_cifar100_bs64_p16_384_eps1e-4_defer/20231226_091558/vis_data/20231226_091558.json',
              ]
    filename_cub=[
            #   'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps1e-2/20240101_002929/vis_data/20240101_002929.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps5e-3/20231231_092736/vis_data/20231231_092736.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps4e-3/20240103_224711/vis_data/20240103_224711.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps3e-3/20231231_092653/vis_data/20231231_092653.json',
            #   'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps25e-4/20231230_081353/vis_data/20231230_081353.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps2e-3/20231230_083004/vis_data/20231230_083004.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps1e-3/20231229_090717/vis_data/20231229_090717.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps1e-4/20231229_090654/vis_data/20231229_090654.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps0/20231228_185803/vis_data/20231228_185803.json',
              ]
    filename_cub_defer=[
            #   'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps1e-2_defer/20240101_024903/vis_data/20240101_024903.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps5e-3_defer/20231231_205423/vis_data/20231231_205423.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps4e-3_defer/20240108_172356/vis_data/20240108_172356.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps3e-3_defer/20240102_090515/vis_data/20240102_090515.json',
            #   'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps25e-4_defer/20231230_162009/vis_data/20231230_162009.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps2e-3_defer/20231230_165231/vis_data/20231230_165231.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps1e-3_defer/20231229_180646/vis_data/20231229_180646.json',
              'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps1e-4_defer/20231229_192028/vis_data/20231229_192028.json',
              ]
    
    filename_oxford_pets=[
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps1e-2/20240103_224412/vis_data/20240103_224412.json',
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps5e-3/20240103_141004/vis_data/20240103_141004.json',
            #   'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps4e-3/20240104_024954/vis_data/20240104_024954.json',
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps3e-3/20240102_213027/vis_data/20240102_213027.json',
            #   'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps25e-4/20231230_081353/vis_data/20231230_081353.json',
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps2e-3/20240101_233511/vis_data/20240101_233511.json',
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps1e-3/20240101_094637/vis_data/20240101_094637.json',
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps1e-4/20240101_122210/vis_data/20240101_122210.json',
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps0/20240101_094449/vis_data/20240101_094449.json',
              ]
    filename_oxford_pets_defer=[
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps1e-2_defer/20240103_141808/vis_data/20240103_141808.json',
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps5e-3_defer/20240103_091741/vis_data/20240103_091741.json',
            #   'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps4e-3_defer/20240104_074943/vis_data/20240104_074943.json',
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps3e-3_defer/20240102_154523/vis_data/20240102_154523.json',
            #   'work_dirs_20240104/vit_config_cub_bs64_p16_384_eps25e-4/20231230_081353/vis_data/20231230_081353.json',
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps2e-3_defer/20240102_090215/vis_data/20240102_090215.json',
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps1e-3_defer/20240101_163511/vis_data/20240101_163511.json',
              'work_dirs_20240104/vit_config_oxford_iii_pets_bs64_p16_384_eps1e-4_defer/20240101_195412/vis_data/20240101_195412.json',
              
              ]
    eps=get_eps(filename_cifar100)
    print(eps)
    # plot_curve_defer(filename_cifar10,filename_cifar10_defer,dataset='cifar10',kind='val')
    # plot_curve_defer(filename_cifar10,filename_cifar10_defer,dataset='cifar10',kind='train')
    # plot_curve_defer(filename_cifar100,filename_cifar100_defer,dataset='cifar100',kind='val')
    # plot_curve_defer(filename_cifar100,filename_cifar100_defer,dataset='cifar100',kind='train')
    plot_curve_defer(filename_cub,filename_cub_defer,dataset='cub',kind='val')
    plot_curve_defer(filename_cub,filename_cub_defer,dataset='cub',kind='train')
    # plot_curve_defer(filename_oxford_pets,filename_oxford_pets_defer,dataset='oxford_pets',kind='val')
    # plot_curve_defer(filename_oxford_pets,filename_oxford_pets_defer,dataset='oxford_pets',kind='train')
    # plot_loss_curve(filename_cifar100,dataset='cifar100',kind='val')
    exit()

    log_dicts=get_log_dicts(filename)[3]
    # print(log_dicts)
    plot_loss(log_dicts)
    plot_accu(log_dicts)

