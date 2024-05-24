import nengo
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
from matplotlib import style
import scienceplots
import seaborn as sns

def generate_white_noise(length, dt, freq, y0=0):
    """ Generates white noise using Nengo. """
    process = nengo.processes.WhiteSignal(period=length * dt, high=freq, y0=y0)
    with nengo.Network() as model:
        noise_node = nengo.Node(process, size_out=1)
        probe = nengo.Probe(noise_node)
    with nengo.Simulator(model) as sim:
        sim.run(length * dt)
    return sim.data[probe].flatten()


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

def qofd(x,order):
    # x=torch.tensor(x)
    n=x.shape[0]
    # tilde_x=torch.zeros_like(torch.tensor(x))
    coor = torch.cos(torch.pi * (torch.arange(n) + 0.5) / n)
    cheby_poly=torch.cos(torch.outer(torch.arange(order).double() , torch.acos(coor)))
    # x=x.unsqueeze(dim=0)
    coefficients=torch.matmul(x,cheby_poly.T)*2/n
    coefficients[...,0]=coefficients[...,0]/2

    tilde_x=torch.matmul(coefficients,cheby_poly)
    return tilde_x

def l_infty(x,sampling_times):

    # x_max=torch.max(torch.abs(x))
    yn_2=x*x/torch.norm(x)/torch.norm(x)
    yn_2=torch.where(yn_2>1,torch.full_like(yn_2,1),yn_2)
    # m=torch.distributions.multinomial.Multinomial(sampling_times,torch.abs(yn) ** 2).sample()/sampling_times
    m=torch.distributions.binomial.Binomial(sampling_times,yn_2).sample()
    m=m/torch.unsqueeze(torch.sum(m,dim=-1),dim=-1)

    new_states = torch.cat(((x+torch.sqrt(m))/torch.sqrt(torch.tensor(2.0)),(x-torch.sqrt(m))/torch.sqrt(torch.tensor(2.0))),dim=-1)
    new_states=(new_states/torch.unsqueeze(torch.norm(new_states,dim=-1),dim=-1))**2
    # outcomes=torch.distributions.multinomial.Multinomial(sampling_times,new_states).sample()

    new_states=torch.where(torch.isnan(new_states) ,torch.full_like(new_states,0),new_states)
    new_states=torch.where(torch.isinf(new_states),torch.full_like(new_states,0),new_states)
    # new_states=torch.where(torch.isinf(yn),torch.full_like(yn,0),yn)
    outcomes=torch.distributions.binomial.Binomial(sampling_times,new_states).sample()
    outcomes=outcomes/torch.unsqueeze(torch.sum(outcomes,dim=-1),dim=-1)
    n_0_i = outcomes[...,:x.shape[-1]]
    sigma_i = torch.where(n_0_i/m > 0.4 , torch.tensor(1.0), torch.tensor(-1.0))
    m = sigma_i * torch.sqrt(m)

    # m=m/m.norm(2)*y_norm 
    m1=torch.norm(x,dim=-1)/torch.norm(m,dim=-1)
    m1=m1.unsqueeze(dim=-1)
    m=m*m1
    m=torch.where(torch.isnan(m),torch.full_like(m,0),m)
    m=torch.where(torch.isinf(m),torch.full_like(m,0),m)
    return m 


def cosine_similarity(x,y):
    return torch.dot(x,y)/torch.norm(x)/torch.norm(y)


def cos_similarity_test(x,mode,order_or_times):
    cos_sim=[]
    for order in order_or_times:
        if mode=='qofd':
            y=qofd(x,order)
            temp=np.dot(y,x)/np.linalg.norm(y)/np.linalg.norm(x)
            cos_sim.append(temp)
        elif mode=='l_inf':
            y=l_infty(x,order)
            temp=np.dot(y,x)/np.linalg.norm(y)/np.linalg.norm(x)
            cos_sim.append(temp)
    return cos_sim


def generate_data(length,dt,freq,y0,repeat_times):
    # length = 10000
    # dt = 0.001
    # freq = 1
    # y0 = 0
    file_pre='test_qofd_0513/random_functions/'
    fname1=file_pre+f'dim={length}_qofd.txt'
    fname2=file_pre+f'dim={length}_q_inf.txt'
    # repeat_times=20
    order_range=[2*i for i in range(1,21)]
    times_range=[2**i for i in range(5,21)]
    # order_range=[2*i for i in range(1,5)]
    # times_range=[2**i for i in range(10,14)]

    for i in range(repeat_times):
        print(i)
        y=generate_white_noise(length,dt,freq,y0)
        y=y/np.linalg.norm(y)
        y=torch.tensor(y)

        qofd_result=cos_similarity_test(y,'qofd',order_range)
        q_inf_result=cos_similarity_test(y,'l_inf',times_range)
        with open(fname1, 'a') as f:
            f.write(f'{qofd_result}\n')
        with open(fname2, 'a') as f:
            f.write(f'{q_inf_result}\n')


def get_results(filename):
  cos_sim_sum=[]
  for f in filename:
    file=open(f,'r')
    cos_sim=[]
    for line in file.readlines():
        data_list = json.loads(line)
        #   temp=np.array(data_list)
        cos_sim.append(data_list)
        #  print(dim,sqrt_p)
    cos_sim_sum.append(cos_sim)
  return np.array(cos_sim_sum)



# def draw_figures_qofd(data,order_range):

#     cos_ave=[np.average(data[:,i]) for i in range(data.shape[1])]
#     cos_std=[np.std(data[:,i]) for i in range(data.shape[1])]
#     plt.scatter(order_range,cos_ave)
#     plt.errorbar(order_range, cos_ave, yerr=cos_std,capsize=3, capthick=2,label=f'length={10000}')
#     plt.show()



# 示例函数，绘制带误差线的散点图
def draw_figures(mode_name, len_range):
    # 计算均值和标准差
    

    # plt.style.use(['science','ieee', 'grid'])
    style.use('seaborn-v0_8-darkgrid')
    plt.style.use(['science', 'grid'])
    
    # 创建图表
    plt.figure(figsize=(8, 6))

    data_all=get_results(mode_name['fname'])
    x_axis=data_all.shape[-1]
    if mode_name['mode']=='qofd':
        x_axis=15

    color_lib=['blue','green','red']
    for r,data in enumerate(data_all):
        cos_ave = [np.average(data[:, i]) for i in range(x_axis)]
        cos_std = [np.std(data[:, i]) for i in range(x_axis)]

        plt.errorbar(mode_name['ot_range'][0:x_axis], cos_ave, yerr=cos_std, fmt='o', color=color_lib[r], 
                    ecolor=color_lib[r], elinewidth=2, capsize=3, capthick=2)
        plt.scatter(mode_name['ot_range'][0:x_axis], cos_ave, color=color_lib[r], s=40)  # s 控制点的大小
        plt.plot(mode_name['ot_range'][0:x_axis], cos_ave,label=f'Dimension = {len_range[r]}',lw=3,color=color_lib[r])
    
    # 设置图表标题和标签
    if mode_name['mode']=='qofd':
        plt.title('Quantum Orthogonal Function Decomposition', fontsize=20)
        plt.xlabel('Decomposition Order', fontsize=16)
        plt.legend(fontsize=16,loc=4)
    elif mode_name['mode']=='l_inf':
        plt.title('$l_{\infty}$ Tomography', fontsize=20)
        plt.xlabel('Sampling Times', fontsize=16)
        plt.xscale('log')
        plt.legend(fontsize=16,loc=2)
    plt.ylabel('Cosine Similarity', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)  # 主刻度标签大小
    mode1=mode_name['mode']
    plt.savefig(f'random_functions_{mode1}.png')
    plt.savefig(f'random_functions_{mode1}.pdf')
    
    # 添加图例
    
    
    # 展示图表
    plt.show()




if __name__=="__main__":
    b=style.available
    print(b)

    # length = 1000000
    # dt = 0.001
    # freq = 0.01
    # y0 = 0
    # repeat_times=20
    # generate_data(length,dt,freq,y0,repeat_times)
    # exit()

    len_range=[10000,100000,1000000]
    file_pre='test_qofd_0513/random_functions/'
    fname1=[file_pre+f'dim={length}_qofd.txt' for length in len_range]
    fname2=[file_pre+f'dim={length}_q_inf.txt' for length in len_range]
    order_range=[2*i for i in range(1,21)]
    times_range=[2**i for i in range(5,21)]
    # data_all=get_results(fname1)
    # mode='qofd'
    qofd_mode={'mode':'qofd','ot_range':order_range,'fname':fname1}
    l_inf_mode={'mode':'l_inf','ot_range':times_range,'fname':fname2}
    # draw_figures(qofd_mode,len_range)
    draw_figures(l_inf_mode,len_range)
    # order=30
    # y1=torch.tensor(y,dtype=float)
    # y2=qofd(y1,order)
    # y2=y2.numpy()
    # sampling_times=10000
    # y3=l_infty(y1,sampling_times)
    # cos_sim=np.dot(y,y2)/np.linalg.norm(y)/np.linalg.norm(y2)
    # cos_sim2=np.dot(y,y3)/np.linalg.norm(y)/np.linalg.norm(y3)
    # print(cos_sim,cos_sim2)
    # plt.plot(range(length),y,label='ori',color='r')
    # # plt.scatter(range(length),y)
    # plt.plot(range(length),y2,label=f'cheby:order={order}',color='b')
    # # plt.scatter(range(length),y)
    # plt.scatter(range(length),y3,label=f'l_inf:times={sampling_times:.2e}',s=2,color='g')
    # plt.title(f'(ori,qofd)={cos_sim},(ori,l_inf)={cos_sim2}')
    # plt.legend(loc=2)
    # plt.show()



