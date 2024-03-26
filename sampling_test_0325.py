
import mmpretrain
from mmpretrain import inference_model
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import torch
from mmpretrain import get_model
from mmengine.config import Config
from mmpretrain import list_models
from mmpretrain import ImageClassificationInferencer
from mmpretrain.models.utils import LinearSampling
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt



def linear_sampling_test():
    m1=LinearSampling(in_features=2,out_features=3,sampling_error=0.1)
    # m2=LinearSampling(in_features=2,out_features=3,sampling_error=0.)
    x=torch.rand(1,4,2,requires_grad=True)
    y=m1(x)
    loss=y.sum()
    loss.backward()
    print(x.grad)
    print(m1.weight.grad)
    print(m1.bias.grad)

    # y2=m2(x)
    # loss2=y2.sum()
    # loss2.backward()
    # print(m2.weight.grad)
    # print(m2.bias.grad)
def npz_to_pth():
    numpy_weights = np.load('imagenet21k_ViT-B_16.npz')
    # 创建一个空的 PyTorch 模型
    # model = YourModel()  # 用你的模型类初始化

    # # 将 NumPy 字典中的权重复制到 PyTorch 模型的 state_dict 中
    # model_state_dict = model.state_dict()

    # for key in model_state_dict.keys():
    #     # 从 NumPy 数组中复制权重
    #     model_state_dict[key] = torch.from_numpy(numpy_weights[key])

    # # 将 state_dict 设置到 PyTorch 模型中
    # model.load_state_dict(model_state_dict)
    # # tensor_weights = torch.from_numpy(numpy_weights['weight_array']).float()
    # torch.save(tensor_weights, 'imagenet21k_ViT-B_16.pth')

def cosine_similarity(x,y):
    return torch.dot(x,y)/torch.norm(x)/torch.norm(y)

def sampling_compare(a,b,sampling_times):
    c=a+b
    # a1=torch.distributions.multinomial.Multinomial(sampling_times,a).sample()/sampling_times
    a1=sampling_vector(a,sampling_times)
    a1=a1/torch.norm(a1)
    # c1=torch.distributions.multinomial.Multinomial(sampling_times,c).sample()/sampling_times
    c1=sampling_vector(c,sampling_times)
    c1=c1/torch.norm(c1)
    c2=a1+b 
    c2=c2/torch.norm(c2)
    x1=cosine_similarity(c,c1)
    x2=cosine_similarity(c,c2)
    return x1,x2

def sampling_vector(y,sampling_times):
    y_norm=torch.norm(y)
    y_abs=y/y_norm*y/y_norm 
    # m=torch.distributions.binomial.Binomial(sampling_times,y_abs).sample()
    m=torch.distributions.multinomial.Multinomial(sampling_times,y_abs).sample()/sampling_times
    m=torch.sqrt(m)
    m=torch.sign(y)*m 
    m=m/m.norm(2)*y_norm 
    return m

def sampling_test():
    dim=100000
    a=torch.rand(dim)*2-1
    a=a/torch.norm(a)
    b=torch.rand(dim)*2-1
    b=b/torch.norm(b)

    sampling_times_range=[2**i for i in range(6,20)]
    y1=[]
    y2=[]
    for sampling_times in sampling_times_range:
        temp1,temp2=sampling_compare(a,b,sampling_times)
        y1.append(temp1)
        y2.append(temp2)
    
    plt.plot(sampling_times_range,y1,label='defer')
    plt.plot(sampling_times_range,y2,label='q resudual')
    plt.xscale('log')
    plt.legend(loc=4)
    plt.show()



def chebyshev_polynomials(x, degree):
    if degree == 0:
        return torch.ones_like(x)
    elif degree == 1:
        return x
    else:
        T_prev_prev = torch.ones_like(x)
        T_prev = x
        for n in range(2, degree + 1):
            T_curr = 2 * x * T_prev - T_prev_prev
            T_prev_prev, T_prev = T_prev, T_curr
        return T_curr

def compute_coefficients(func, degree, num_points=10000):
    x = torch.linspace(-1, 1, steps=num_points)
    dx = x[1] - x[0]
    coefficients = []
    
    for n in range(degree + 1):
        Tn = chebyshev_polynomials(x, n)
        integrand = func(x) * Tn / torch.sqrt(1 - x**2)
        # 使用梯形规则进行数值积分
        cn = integrand.sum() * dx
        if n == 0:
            cn = cn / torch.pi
        else:
            cn = cn * 2 / torch.pi
        coefficients.append(cn)
        
    return coefficients

def approximate_function(coefficients, x):
    approximation = torch.zeros_like(x)
    for n, cn in enumerate(coefficients):
        approximation += cn * chebyshev_polynomials(x, n)
    return approximation






if __name__=="__main__":
    # npz_to_pth()
    # exit()
    # sampling_times=10**2
    dim=100
    a=torch.rand(dim)*2-1
    a=a/torch.norm(a)
    b=torch.rand(dim)*2-1
    b=b/torch.norm(b)

    coef=compute_coefficients(func, 10, num_points=10000)





    sampling_times_range=[2**i for i in range(6,20)]
    y1=[]
    y2=[]
    for sampling_times in sampling_times_range:
        temp1,temp2=sampling_compare(a,b,sampling_times)
        y1.append(temp1)
        y2.append(temp2)
    
    plt.plot(sampling_times_range,y1,label='defer')
    plt.plot(sampling_times_range,y2,label='q resudual')
    plt.xscale('log')
    plt.legend(loc=4)
    plt.show()
    # # print(a)
    # a1=torch.distributions.multinomial.Multinomial(sampling_times,a).sample()/sampling_times
    # a1=a1/torch.norm(a1)
    
    # c=a+b
    # c=c/torch.norm(c)
    # c1=torch.distributions.multinomial.Multinomial(sampling_times,c).sample()/sampling_times
    # c1=c1/torch.norm(c1)
    # c2=a1+b 
    # c2=c2/torch.norm(c2)

    # x1=cosine_similarity(c,c1)
    # x2=cosine_similarity(c,c2)
    # print(x1,x2)

    # print(c1)
    # a=torch.rand(2,3,4)
    # b=torch.unsqueeze(a,dim=1)
    # print(a.size(),b.size())
    # linear_sampling_test()
    exit()
    m=nn.Linear(3,4)
    # inference_test_1()
    x=torch.rand(2,4,3)
    y=m(x)
    print(y,y.shape)

    exit()


    inference_test()

    # predict = inference_model('vit-base-p32_in21k-pre_3rdparty_in1k-384px', 'demo/bird.JPEG')
    # print(predict['pred_class'])
    # print(predict['pred_score'])
    # train_test()
    # exit()


    # model = get_model('vit-base-p32_in21k-pre_3rdparty_in1k-384px', pretrained=True)
    # # inputs = torch.rand(1, 3, 224, 224)
    # inputs = torch.rand(1, 3, 224, 224)
    # out = model(inputs)
    # print(type(out))
    # # To extract features.
    # feats = model.extract_feat(inputs)
    # print(type(feats))