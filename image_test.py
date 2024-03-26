import matplotlib.pyplot as plt
import numpy as np
import scienceplots

import seaborn as sns

def classical_vit(n,eps,delta,d):
    return n*d*(n+d)*np.log2(1/eps)

def q_forward(n,eps,delta,d):
    return d*d*np.log2(n)/eps/delta/delta 

def q_back(n,eps,delta,d):
    return d*d*np.sqrt(n)*np.log2(n)/eps/delta/delta 


if __name__=="__main__":
    n_range=[2**i for i in range(10,40)]
    eps=0.001
    delta=0.001
    d=1000
    c=[classical_vit(n,eps,delta,d) for n in n_range]
    qford=[q_forward(n,eps,delta,d) for n in n_range]
    qback=[q_back(n,eps,delta,d) for n in n_range]
    plt.style.use(['science','ieee'])

    plt.plot(n_range,c,label='classical')
    plt.plot(n_range,qford,label='quantum forward')
    plt.plot(n_range,qback,label='quantum backpropagation')
    plt.title('eps=0.001,delta=0.001,d=1000')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('complexity')

    plt.legend(loc=4)
    plt.savefig('zzz.jpg')

    # plt.show()