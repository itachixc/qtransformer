import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch



def get_succ_prob(filename):
  file=open(filename,'r')
  cos_sim=[]
  for line in file.readlines():
      temp=float(line[0:-1])
      cos_sim.append(temp)
    #  print(dim,sqrt_p)
  return np.average(cos_sim)










if __name__=="__main__":
    # 示例向量
    # vector = np.array([np.cos(i*0.1) for i in range(100)])
    # vector = np.array([i for i in range(100)])/100
    # a = torch.tensor([1, 2, 3])
    # # 计算外积
    # outer_product = torch.outer(a, b)
    # print("外积:\n", outer_product)
    # exit()
    dataset='cifar100'
    order_range=[2,5,10,20,30,40,50]
    cos_sim_qofd=[]
    cos_sim_inf=[]
    
    for order in order_range:
        fname1=f'test_qofd_0513/{dataset}/{dataset}_{order}_qofd.txt'
        fname2=f'test_qofd_0513/{dataset}/{dataset}_{order}_inf.txt'
        temp1=get_succ_prob(fname1)
        temp2=get_succ_prob(fname2)
        cos_sim_qofd.append(temp1)
        cos_sim_inf.append(temp2)
    plt.plot(order_range,cos_sim_qofd,label='qofd')
    plt.scatter(order_range,cos_sim_qofd)
    plt.plot(order_range,cos_sim_inf,label='inf')
    plt.scatter(order_range,cos_sim_inf)
    plt.xlabel('order')
    plt.ylabel('cosine similarity')
    plt.legend(loc=4)
    plt.title(dataset)
    plt.show()

    exit()

    # 示例使用
    # 示例使用
    m = 2 # 示例数据长度
    p=4
    n = 10   # 第二维度的大小
    y = torch.rand(m, p,n)  # 创建一个随机的多维torch.tensor
    decomposition_order = 10  # 分解阶数

    z = chebyshev_dec_test_torch_multidim(y, decomposition_order)
    # print(z.shape)
    # print(y)
    # print(z)
    # print(torch.norm(y-z))

    exit()

    file_name='./test_cheby_0326/cifar10/cheby_shev_10.pt'
    image_array=load_data(file_name)

   
    
    
    print(image_array.shape)  # 打印数组的形状，通常是(高度, 宽度, 通道数)
    # exit()

    vector=image_array[0,:,3]
    vector=vector.flatten()
    vector=vector/np.max(vector)
    m = len(vector)
    print(m)

    order_range=[10*i for i in range(1,20)]
    y=[]
    for i in order_range:
        z=chebyshev_dec_test(vector,i)
        cos_sim=cosine_similarity(z,vector)
        y.append(cos_sim)
    plt.plot(order_range,y)
    plt.show()
    # print(cos_sim)
    exit()
    # rand_mat=np.random.random((m,m))
    # vector=np.dot(rand_mat,vector)
    # vector=vector/np.max(vector)
    # vector=np.random.random(50)
    N=40
    # 将向量索引映射到[-1, 1]区间
    
    x = np.cos(np.pi * (np.arange(m) + 0.5) / m)

    # 计算切比雪夫系数
    
    coefficients = np.zeros(N)
    for n in range(N):
        Tn = np.cos(n * np.arccos(x))  # 切比雪夫多项式的值
        coefficients[n] = np.dot(vector, Tn) * 2 / m
    coefficients[0] /= 2

    print("切比雪夫系数:", coefficients)
    y=reconstruct(x,coefficients)
    print(y)
    # plt.plot(abs(coefficients))

    plt.plot(x,vector,label='func')
    plt.scatter(x,vector)
    plt.plot(x,y,label="cheby")
    plt.scatter(x,y)
    plt.legend(loc=1)
    plt.show()