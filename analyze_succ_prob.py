import argparse
import os
import re
from itertools import groupby
import scienceplots

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mmpretrain.utils import load_json_log






def get_succ_prob(filename):
  file=open(filename,'r')
  succ_prob={}
  for line in file.readlines():
     temp=line.split(' ')
     dim=int(temp[0])
     sqrt_p=float(temp[1])
     if dim in succ_prob.keys():
        succ_prob[dim].append(sqrt_p)
     else:
        if dim>5000:
          succ_prob[dim]=[sqrt_p]
        
    #  print(dim,sqrt_p)
  return succ_prob

def get_main_prob(filename,r_dim):
  file=open(filename,'r')
  succ_prob=[]
  # r_dim=((r_size/16)**2+1)*768
  file_name_split=filename.split('_')
  for line in file.readlines():
    temp=line.split(' ')
    dim=int(temp[0])
    sqrt_p=float(temp[1])
    if dim==r_dim:
      if file_name_split[-1]=='backpropagation.txt' and file_name_split[-2]=='block-encoding':
        if sqrt_p<0.05:
          succ_prob.append(sqrt_p)
      else:
        succ_prob.append(sqrt_p)  
  return succ_prob
     
def plot_curve_main_succ_r(dataset,r_size_range,mode1_r,mode2_r):
  x=[((r_size/16)**2+1)*768 for r_size in r_size_range]
  for mode1 in mode1_r:
    for mode2 in mode2_r:
      y=[]
      error_bar=[]
      for r_size in r_size_range:
        filename=f'prob_results/{dataset}/{dataset}_{r_size}_{mode1}_{mode2}.txt'
        r_dim=((r_size/16)**2+1)*768
        succ_prob=get_main_prob(filename,r_dim)
        y.append(np.average(succ_prob))
        temp1=np.std(succ_prob)
        error_bar.append(temp1)
      plt.scatter(x,y,label=f'{mode1}-{mode2}')
      plt.errorbar(x, y, yerr=error_bar,capsize=3, capthick=2)
  plt.yscale('log')
  plt.xscale('log')
  plt.legend(loc=1)
  plt.show()


def plot_curve_main_succ_fit(dataset,r_size_range,mode1='qdac',mode2='backpropagation'):
  x=[((r_size/16)**2+1)*768 for r_size in r_size_range]
  y=[]
  for r_size in r_size_range:
    filename=f'prob_results_0122/{dataset}/{dataset}_{r_size}_{mode1}_{mode2}.txt'
    r_dim=((r_size/16)**2+1)*768
    succ_prob=get_main_prob(filename,r_dim)
    y.append(np.average(succ_prob))
  log_x=np.array([np.log10(i) for i in x])
  log_y=np.array([np.log10(i) for i in y])
  slope,intercept=np.polyfit(log_x,log_y,1)
  fit_logy=[slope*i+intercept for i in log_x]
  print(slope,intercept)
  plt.scatter(log_x,log_y)
  plt.plot(log_x,fit_logy,color='red')
  plt.show()

def plot_curve_main_succ_mata(dataset,r_size_range,mode1_r):
  # x=[((r_size/16)**2+1)*768 for r_size in r_size_range]
  for mode1 in mode1_r:
      error_bar=[]
      y=[]
      x=[]
      for r_size in r_size_range:
        filename=f'prob_results_0123/{dataset}/{dataset}_{r_size}_a_{mode1}_forward.txt'
        if mode1=='qdac':
          r_dim=((r_size/16)**2+1)**2
        else:
          r_dim=((r_size/16)**2+1)*64
        x.append(r_dim)
        succ_prob=get_main_prob(filename,r_dim)
        y.append(np.average(succ_prob))
        temp1=np.std(succ_prob)
        error_bar.append(temp1)
      plt.scatter(x,y,label=f'{mode1}-forward')
      plt.errorbar(x, y, yerr=error_bar,capsize=3, capthick=2)
  plt.yscale('log')
  plt.xscale('log')
  plt.legend(loc=1)
  plt.show()



def plot_curve_main_succ(dataset,r_size_range,mode1,mode2):
  x=[((r_size/16)**2+1)*768 for r_size in r_size_range]
  y=[]
  for r_size in r_size_range:
    filename=f'prob_results/{dataset}/{dataset}_{r_size}_{mode1}_{mode2}.txt'
    succ_prob=get_main_prob(filename,r_size)
    y.append(np.average(succ_prob))
  plt.scatter(x,y)
  # plt.yscale('log')
  plt.xscale('log')
  plt.show()

     
def plot_curve_succ(dataset,r_size_range,mode1,mode2):

  for r_size in r_size_range:
    filename=f'prob_results/{dataset}/{dataset}_{r_size}_{mode1}_{mode2}.txt'
    succ_prob=get_succ_prob(filename)
    x=[keys for keys in succ_prob]
    y=[]
    error_bar=[]
    for key in succ_prob:
        temp=np.average(succ_prob[key])
        temp1=np.std(succ_prob[key])
        y.append(temp)
        error_bar.append(temp1)
    plt.errorbar(x, y, yerr=error_bar,capsize=3, capthick=2)
    plt.scatter(x,y)
  # plt.yscale('log')
  # plt.xscale('log')
  plt.show()
  






if __name__=="__main__":
    
    # plot_loss_curve(filename_cifar100,dataset='cifar100',kind='val')
    # dataset='cifar10'
    dataset='cub'
    # dataset='oxford_iii_pets'
    # dataset='cifar100'
    # r_size_range=[640,576,512,448,384,320,256,192,128,64]
    r_size_range=[640,576,512,448,384,320,256,192,128,64]
    # mode1='qdac'
    # # mode1='block-encoding'
    # mode2='forward'
    # # mode2='backpropagation'
    # # plot_curve_succ(dataset,r_size_range,mode1,mode2)
    # plot_curve_main_succ(dataset,r_size_range,mode1,mode2)


    mode1_r=['qdac','block-encoding']
    mode2_r=['forward','backpropagation']
    # plot_curve_main_succ_r(dataset,r_size_range,mode1_r,mode2_r)

    plot_curve_main_succ_mata(dataset,r_size_range,mode1_r)

    # plot_curve_main_succ_fit(dataset,r_size_range,mode1='qdac',mode2='backpropagation')




    # r_size=384
    # filename=f'prob_results/{dataset}/{dataset}_{r_size}_qdac_backpropagation.txt'
    # succ_prob=get_succ_prob(filename)
    # print(succ_prob.keys())
    exit()

    log_dicts=get_log_dicts(filename)[3]
    # print(log_dicts)
    plot_loss(log_dicts)
    plot_accu(log_dicts)

