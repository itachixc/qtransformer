import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def reconstruct(x,coefficients):
    y=np.zeros(x.shape[0])
    for n,coef in enumerate(coefficients):
        y=y+coef*np.cos(n * np.arccos(x))
    return y


def chebyshev_dec_test(y,decomposition_order):
    m = len(y)
    x = np.cos(np.pi * (np.arange(m) + 0.5) / m)
    # 计算切比雪夫系数
    coefficients = np.zeros(decomposition_order)
    for n in range(decomposition_order):
        Tn = np.cos(n * np.arccos(x))  # 切比雪夫多项式的值
        coefficients[n] = np.dot(y, Tn) * 2 / m
    coefficients[0] /= 2
    z=np.zeros(x.shape[0])
    for n,coef in enumerate(coefficients):
        z=z+coef*np.cos(n * np.arccos(x))

    plt.plot(x,y,label='func')
    plt.scatter(x,y)
    plt.plot(x,z,label="cheby")
    plt.scatter(x,z)
    plt.legend(loc=1)
    plt.show()
    return z

import torch

def chebyshev_dec_test_torch(y, decomposition_order):
    m = len(y)
    x = torch.cos(torch.pi * (torch.arange(m).float() + 0.5) / m)
    # 计算切比雪夫系数
    coefficients = torch.zeros(decomposition_order)
    for n in range(decomposition_order):
        Tn = torch.cos(n * torch.acos(x))  # 切比雪夫多项式的值
        coefficients[n] = torch.dot(y, Tn) * 2 / m
    coefficients[0] /= 2
    z = torch.zeros(x.shape[0])
    for n, coef in enumerate(coefficients):
        z = z + coef * torch.cos(n * torch.acos(x))
    return z








def test():
    image_path = 'demo/bird.JPEG'  # 替换为你的图片路径
    image = Image.open(image_path)
    # 将图片转换为Numpy数组
    image_array = np.array(image)
    print(image_array.shape)  # 打印数组的形状，通常是(高度, 宽度, 通道数)
    # exit()
    vector=image_array[:,100,0]
    vector=vector/np.max(vector)
    m = len(vector)

    chebyshev_dec_test(vector,40)


def load_data(file_name):
    loaded_tensors = torch.load(file_name)
    load_array=loaded_tensors.detach().numpy()
    return load_array

def cosine_similarity(x,y):
    return np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)

def chebyshev_dec_test_torch_multidim(y, decomposition_order):
    
    x = torch.cos(torch.pi * (torch.arange(y.shape[-1]).float() + 0.5) / y.shape[-1])
    # chebyshev polynomial
    cheby_poly=torch.cos(torch.outer(torch.arange(decomposition_order) , torch.acos(x)))
    a=torch.matmul(cheby_poly,cheby_poly.T)
    print(a)
    # coefficients
    coefficients=torch.matmul(y,cheby_poly.T)*2/y.shape[-1]
    coefficients[...,0]=coefficients[...,0]/2
    # restructed y
    restructed_y=torch.matmul(coefficients,cheby_poly)
   
    return restructed_y


if __name__=="__main__":
    # 示例向量
    # vector = np.array([np.cos(i*0.1) for i in range(100)])
    # vector = np.array([i for i in range(100)])/100
    # a = torch.tensor([1, 2, 3])
    # # 计算外积
    # outer_product = torch.outer(a, b)
    # print("外积:\n", outer_product)
    # exit()
    test()
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