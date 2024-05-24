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


def l_infty(x,eps):

    # x_max=torch.max(torch.abs(x))
    sampling_times=int(36*np.log2(x.shape[0])/eps/eps)
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


def cosine_similarity_test(x,y,eps_range,mode):
    cos_sim=[]
    z=x+y
    for eps in eps_range:
        if mode=='origin':
            y_tilde=l_infty(y,eps)
            z_tilde=x+y_tilde
            temp=np.dot(z,z_tilde)/np.linalg.norm(z)/np.linalg.norm(z_tilde)
            cos_sim.append(temp)
        elif mode=='defer':
            z_tilde=l_infty(z,eps)
            temp=np.dot(z,z_tilde)/np.linalg.norm(z)/np.linalg.norm(z_tilde)
            cos_sim.append(temp)
    return cos_sim


def generate_data(length,dt,freq,y0,repeat_times):
    # length = 10000
    # dt = 0.001
    # freq = 1
    # y0 = 0
    file_pre='test_qofd_0513/rf_multiple_reuse/'
    fname1=file_pre+f'dim={length}_.txt'
    fname2=file_pre+f'dim={length}_defer.txt'
    # repeat_times=20
    # order_range=[2*i for i in range(1,5)]
    # times_range=[2**i for i in range(10,14)]

    for i in range(repeat_times):
        print(i)
        x=generate_white_noise(length,dt,freq,y0)
        x=x/np.linalg.norm(x)
        x=torch.tensor(x)
        y=generate_white_noise(length,dt,freq,y0)
        y=y/np.linalg.norm(y)
        y=torch.tensor(y)
        eps_range=[2**(-i) for i in range(2,10)]

        result=cosine_similarity_test(x,y,eps_range,'origin')
        defer_result=cosine_similarity_test(x,y,eps_range,'defer')
        with open(fname1, 'a') as f:
            f.write(f'{result}\n')
        with open(fname2, 'a') as f:
            f.write(f'{defer_result}\n')


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
def draw_figures(fname1,fname2, eps_range):
    # 计算均值和标准差
    

    # plt.style.use(['science','ieee', 'grid'])
    style.use('seaborn-v0_8-darkgrid')
    plt.style.use(['science', 'grid'])
    
    # 创建图表
    plt.figure(figsize=(8, 6))

    data_1=get_results(fname1)
    data_2=get_results(fname2)
    # x_axis=data_all.shape[-1]
    # if mode_name['mode']=='qofd':
    #     x_axis=15
    x_axis=len(eps_range)
    label=['$10^5$','$10^6$','$10^7$']
    # color_lib=['blue','green','red']
    color_lib=['b','g','r','c','m','k','y','purple']
    for r in range(data_1.shape[0]):
        cos_ave_origin = [np.average(data_1[r,:, i]) for i in range(x_axis)]
        cos_std_origin = [np.std(data_1[r,:, i]) for i in range(x_axis)]

        plt.errorbar(eps_range, cos_ave_origin, yerr=cos_std_origin, fmt='o', color=color_lib[r], 
                    ecolor=color_lib[r], elinewidth=2, capsize=3, capthick=2)
        plt.scatter(eps_range, cos_ave_origin, color=color_lib[r], s=40)  # s 控制点的大小
        plt.plot(eps_range, cos_ave_origin,label=f'Multiple, n = {label[r]}',lw=3,color=color_lib[r])

        cos_ave_defer = [np.average(data_2[r,:, i]) for i in range(x_axis)]
        cos_std_defer = [np.std(data_2[r,:, i]) for i in range(x_axis)]

        plt.errorbar(eps_range, cos_ave_defer, yerr=cos_std_defer, fmt='o', color=color_lib[r], 
                    ecolor=color_lib[r], elinewidth=2, capsize=3, capthick=2)
        plt.scatter(eps_range, cos_ave_defer, color=color_lib[r], s=40)  # s 控制点的大小
        plt.plot(eps_range, cos_ave_defer,label=f'Single, n = {label[r]}',lw=3,color=color_lib[r],linestyle='--')
    
    # 设置图表标题和标签
    
    
    # plt.title('$l_{\infty}$ Tomography', fontsize=20)
    plt.xlabel('Tomography Error', fontsize=16)
    plt.xscale('log')
    plt.xlim(0.003,0.3)
    # plt.yscale('log')
    plt.legend(fontsize=13,loc=3)
    plt.ylabel('Cosine Similarity', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)  # 主刻度标签大小
    plt.savefig(f'rf_mr.png')
    plt.savefig(f'rf_mr.pdf')
    
    # 添加图例
    
    
    # 展示图表
    plt.show()




if __name__=="__main__":
    b=style.available
    print(b)

    # length = 10000000
    # dt = 0.001
    # freq = 0.001
    # y0 = 0
    # repeat_times=20
    # generate_data(length,dt,freq,y0,repeat_times)
    # exit()

    len_range=[100000,1000000,10000000]
    file_pre='test_qofd_0513/rf_multiple_reuse/'
    fname1=[file_pre+f'dim={length}_.txt' for length in len_range]
    fname2=[file_pre+f'dim={length}_defer.txt' for length in len_range]
    eps_range=[2**(-i) for i in range(2,8)]
    # data_all=get_results(fname1)
    # mode='qofd'
    # origin_mode={'mode':'qofd','ot_range':order_range,'fname':fname1}
    # defer_mode={'mode':'l_inf','ot_range':times_range,'fname':fname2}
    # draw_figures(qofd_mode,len_range)
    draw_figures(fname1,fname2,eps_range)
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



