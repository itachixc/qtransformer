import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

if __name__=="__main__":
    # 示例向量
    # vector = np.array([np.cos(i*0.1) for i in range(100)])
    # vector = np.array([i for i in range(100)])/100

    
    import numpy as np

    # 读取图片
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