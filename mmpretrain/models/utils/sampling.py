
from typing import Union
from typing import Any, Dict, Optional
import warnings
import torch.nn as nn
import torch
from torch.distributions import multinomial
import numpy as np
from functools import partial

import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

from mmengine.model import BaseModule

import datetime

import torch.fft as fft
from torch import Tensor
from mmengine.utils import digit_version
from mmcv.cnn.bricks.drop import build_dropout
from .layer_scale import LayerScale

if torch.__version__ == 'parrots':
    TORCH_VERSION = torch.__version__
else:
    # torch.__version__ could be 1.3.1+cu92, we only need the first two
    # for comparison
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])

def obsolete_torch_version(torch_version, version_threshold) -> bool:
    return torch_version == 'parrots' or torch_version <= version_threshold


# Success probability test
compute_success_prob=False
compute_success_prob_block=False
compute_success_a_qdac=False
r_size=640
dataset='cifar100'
dataset='cub'
dataset='oxford_iii_pets'
test_cheby=False
label1=0

is_compare=False
# order=20
file_pre='test_qofd_0513'


def scaled_dot_product_attention_pyimpl(query,
                                        key,
                                        value,
                                        attn_mask=None,
                                        dropout_p=0.,
                                        scale=None,
                                        is_causal=False):
    scale = scale or query.size(-1)**0.5
    if is_causal and attn_mask is not None:
        attn_mask = torch.ones(
            query.size(-2), key.size(-2), dtype=torch.bool).tril(diagonal=0)
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf'))

    attn_weight = query @ key.transpose(-2, -1) / scale
    if attn_mask is not None:
        attn_weight += attn_mask
    sqrt_a=torch.exp(attn_weight*0.5)
    b_j=torch.sum(torch.exp(attn_weight),dim=-1)
    b_j=torch.unsqueeze(b_j,dim=-1)
    b_j=b_j.repeat(1,1,1,b_j.size()[-2])
    sqrt_a_bj=torch.div(sqrt_a,b_j)
    # attn_weight_temp=torch.mul(sqrt_a,sqrt_a_bj)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # test=torch.norm(attn_weight_temp-attn_weight)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    if compute_success_a_qdac:
        temp=torch.squeeze(torch.flatten(attn_weight,2,-1))
        y_size=temp.size()[1]
        norm_a_pie=torch.norm(temp,dim=1).cpu().detach().numpy()
        norm2=torch.norm(temp,float('inf'),dim=1).cpu().detach().numpy()
        temp1=torch.squeeze(torch.flatten(sqrt_a,2,-1))
        norm_sqrt_a=torch.norm(temp1,float('inf'),dim=1).cpu().detach().numpy()
        temp2=torch.squeeze(torch.flatten(sqrt_a_bj,2,-1))
        norm_sqrt_abj=torch.norm(temp2,float('inf'),dim=1).cpu().detach().numpy()
        prob_dac=[norm_a_pie[i]/norm_sqrt_a[i]/norm_sqrt_abj[i]/np.sqrt(y_size) for i in range(temp.size()[0])]
        filename_qdac=f'{file_pre}/{dataset}/{dataset}_{r_size}_a_qdac_forward.txt'
        with open(filename_qdac, 'a') as f:
            for i in range(temp.size()[0]):
                if norm_a_pie[i]>1e-6:
                    prob_dac=norm_a_pie[i]/norm2[i]/np.sqrt(y_size)
                    f.write(f'{y_size} {prob_dac}\n') 
        norm_v=torch.squeeze(torch.norm(value,dim=[2,3]).cpu()).detach().numpy()
        x=attn_weight @ value
        norm_x=torch.squeeze(torch.norm(x,dim=[2,3]).cpu()).detach().numpy()
        filename_block=f'{file_pre}/{dataset}/{dataset}_{r_size}_a_block-encoding_forward.txt'
        with open(filename_block, 'a') as f:
            y_dim=value.size()[-1]*value.size()[-2]
            for i in range(value.size()[1]):
                if norm_x[i]>1e-6:
                    prob_dac=norm_x[i]/norm_a_pie[i]/norm_v[i]
                    f.write(f'{y_dim} {prob_dac}\n') 
    return attn_weight @ value


# if digit_version(torch.__version__) >= digit_version('2.0.0'):
#     scaled_dot_product_attention = F.scaled_dot_product_attention
# else:
#     scaled_dot_product_attention = scaled_dot_product_attention_pyimpl


scaled_dot_product_attention = scaled_dot_product_attention_pyimpl



class NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, new_shape: tuple) -> torch.Tensor:
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple:
        shape = ctx.shape
        return NewEmptyTensorOp.apply(grad, shape), None
# Sampling process: l_infinity tomography or chebyshev decomposition
class SamplingBlock(nn.Module):
    def __init__(self,sampling_error=0.,sampling_mode=1,sampling_order=100):
        super(SamplingBlock, self).__init__()
        self.sampling_mode=sampling_mode
        self.sampling_error=sampling_error
        self.sampling_order=sampling_order
    def forward(self,x):
        # chebyshev_dec_test(x,400)
        if test_cheby:
            global label1
            torch.save(x,f'{file_pre}/{dataset}/cheby_shev_{label1}.pt')
            label1+=1
        if self.sampling_mode==0:
            # print('mode 0')
            return SamplingBlockAutoGrad.apply(x,self.sampling_error)
        elif self.sampling_mode==2:
            # print('mode 0')
            return SamplingBlockComAutoGrad.apply(x,self.sampling_order)
        else:
            # print('mode 1')
            x_max=torch.max(torch.abs(x))
            if x_max<1e-10:
                return x
            else:
                # print(x.shape)
                # print(self.sampling_order)
                x=x.transpose(-1,-2)/x_max
                coor = torch.cos(torch.pi * (torch.arange(x.shape[-1]).float() + 0.5) / x.shape[-1])
                # chebyshev polynomial
                cheby_poly=torch.cos(torch.outer(torch.arange(self.sampling_order) , torch.acos(coor))).to(x.device)
                # coefficients
                coefficients=torch.matmul(x,cheby_poly.T)*2/x.shape[-1]
                coefficients[...,0]=coefficients[...,0]/2
                # restructed y
                restructed_x=torch.matmul(coefficients,cheby_poly)

                # x=x.transpose(-1,-2)/x_max
                # coor = torch.cos(torch.pi * (torch.arange(x.shape[-1]).float() + 0.5) / x.shape[-1])
                # # chebyshev polynomial
                # cheby_poly=torch.cos(torch.outer(torch.arange(x.shape[-1]) , torch.acos(coor))).to(x.device)
                # # coefficients
                # coefficients=torch.matmul(x,cheby_poly.T)*2/x.shape[-1]
                # coefficients[...,0]=coefficients[...,0]/2

                # # abs_coef=torch.abs(coefficients)
                # _,indices=torch.topk(torch.abs(coefficients), self.sampling_order, dim=-1)
                # sizes = list(x.shape[:-1])
                # meshgrids = torch.meshgrid([torch.arange(size, device=x.device) for size in sizes], indexing='ij')
                # expanded_grids = [grid.unsqueeze(-1).expand(*sizes, self.sampling_order) for grid in meshgrids]
                # selected_values = coefficients[(*expanded_grids, indices)]
                # select_cheby=cheby_poly[indices,:]

                # batch_shape = selected_values.shape[:-1]
                # j = selected_values.shape[-1]
                # k = select_cheby.shape[-1]

                # # 创建结果张量
                # restructed_x = torch.zeros(*batch_shape, k, device=selected_values.device)

                # # 逐步计算
                # for n in range(j):
                #     restructed_x += selected_values[..., n].unsqueeze(-1) * select_cheby[..., n, :]


                # restruct_x=torch.matmul(selected_values,select_cheby)
                # restructed_x=torch.einsum('...j,...jk->...k', selected_values, select_cheby)

                # x0=x[0,100,:].cpu().detach().numpy()
                # y0=restructed_x[0,100,:].cpu().detach().numpy()
                # c=np.dot(x0,y0)/np.linalg.norm(x0)/np.linalg.norm(y0)
                # plt.scatter(range(len(x0)), x0,label='origin')
                # plt.plot(y0,label='reconstruct')
                # plt.title(f'(origin,reconstruct)={c}')
                # plt.legend(loc=1)
                # plt.show()
                # plt.savefig('test0513.png')
                # restructed y
                # restructed_x=torch.matmul(coefficients,cheby_poly)

                if is_compare and restructed_x.shape[-1]>500:
                    filename_block=f'{file_pre}/{dataset}/{dataset}_{self.sampling_order}_qofd.txt'
                    with open(filename_block, 'a') as f:
                        cos_sim=torch.sum(x*restructed_x)/torch.norm(x)/torch.norm(restructed_x)
                        f.write(f'{cos_sim}\n')
                        print(cos_sim)
                # print(torch.norm(restructed_x-x))
                restructed_x=restructed_x.transpose(-1,-2)*x_max

                return restructed_x

        

class SamplingBlockAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,sampling_error):
        # 在forward方法中，保存input，供backward使用
        ctx.save_for_backward(x,Tensor([sampling_error]))
        if sampling_error != 0:
            y=torch.squeeze(torch.flatten(x,0,x.dim()-1),0)
            sampling_times=int(36*np.log2(y.shape[0])/sampling_error/sampling_error)
            y_norm=y.norm(2)
            if y_norm>1e-8 and y_norm<1e13:
                y_abs=y/y_norm*y/y_norm 
                m=torch.distributions.binomial.Binomial(sampling_times,y_abs).sample()
                m=torch.sqrt(m)
                m=torch.sign(y)*m 
                m=m/m.norm(2)*y_norm 
                m=torch.reshape(m,x.shape)
                return m
            elif y_norm>1e13:
                return x
            else:
                return x-x 
        else:
            return x

    @staticmethod
    def backward(ctx, x):
        # 在backward方法中，获取保存的input
        input, sampling_error= ctx.saved_tensors
        if sampling_error != 0:
            y=torch.squeeze(torch.flatten(x,0,x.dim()-1),0)
            if compute_success_prob:
                y_max=torch.norm(y,float('inf')).cpu()
                y_f=torch.norm(y).cpu()
                y_size=y.size()[0]
                if y_max>1e-6:
                    sqrt_p=(y_f/y_max/np.sqrt(y_size)).numpy()
                    filename=f'{file_pre}/{dataset}/{dataset}_{r_size}_qdac_backpropagation.txt'
                    with open(filename, 'a') as f:
                        f.write(f'{y_size} {sqrt_p}\n')
                    # pass
            sampling_times=int(np.log2(y.shape[0])/sampling_error/sampling_error)
            y_norm=y.norm(2)
            if y_norm>1e-8 and y_norm<1e13:
                y_abs=y/y_norm*y/y_norm 
                m=torch.distributions.binomial.Binomial(sampling_times,y_abs).sample()
                m=torch.sqrt(m)
                m=torch.sign(y)*m 
                m=m/m.norm(2)*y_norm 
                m=torch.reshape(m,x.shape)
                return m,None
            else:
                return x-x, None
        else: 
            return x, None
        
class SamplingBlockComAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,sampling_order):
        # 在forward方法中，保存input，供backward使用
        ctx.save_for_backward(x,Tensor([sampling_order]))
        if sampling_order != 0:
            x_max=torch.max(torch.abs(x))
            y=x.transpose(-1,-2)/x_max
            yn=y/torch.unsqueeze(torch.norm(y,dim=-1),dim=-1)
            yn=torch.where(torch.isnan(yn),torch.full_like(yn,0),yn)
            yn=torch.where(torch.isinf(yn),torch.full_like(yn,0),yn)
            
            y_norm=y.norm(2)
            # y=torch.squeeze(torch.flatten(x,0,x.dim()-1),0)
            sampling_times=int(sampling_order*100)
            if y_norm>1e-8 and y_norm<1e13:
                # y_abs=y/y_norm*y/y_norm 
                # yn2=torch.abs(yn) ** 2
                yn_2=yn*yn
                yn_2=torch.where(yn_2>1,torch.full_like(yn_2,1),yn_2)
                # m=torch.distributions.multinomial.Multinomial(sampling_times,torch.abs(yn) ** 2).sample()/sampling_times
                m=torch.distributions.binomial.Binomial(sampling_times,yn_2).sample()
                m=m/torch.unsqueeze(torch.sum(m,dim=-1),dim=-1)

                new_states = torch.cat(((yn+torch.sqrt(m))/torch.sqrt(torch.tensor(2.0)),(yn-torch.sqrt(m))/torch.sqrt(torch.tensor(2.0))),dim=-1)
                new_states=(new_states/torch.unsqueeze(torch.norm(new_states,dim=-1),dim=-1))**2
                # outcomes=torch.distributions.multinomial.Multinomial(sampling_times,new_states).sample()

                new_states=torch.where(torch.isnan(new_states) ,torch.full_like(new_states,0),new_states)
                new_states=torch.where(torch.isinf(new_states),torch.full_like(new_states,0),new_states)
                # new_states=torch.where(torch.isinf(yn),torch.full_like(yn,0),yn)
                outcomes=torch.distributions.binomial.Binomial(sampling_times,new_states).sample()
                outcomes=outcomes/torch.unsqueeze(torch.sum(outcomes,dim=-1),dim=-1)
                n_0_i = outcomes[...,:y.shape[-1]]
                sigma_i = torch.where(n_0_i/m > 0.4 , torch.tensor(1.0), torch.tensor(-1.0))
                m = sigma_i * torch.sqrt(m)

                # m=m/m.norm(2)*y_norm 
                m1=torch.norm(y,dim=-1)/torch.norm(m,dim=-1)
                m1=m1.unsqueeze(dim=-1)
                m=m*m1*x_max
                m=m.transpose(-1,-2)
                m=torch.where(torch.isnan(m),torch.full_like(m,0),m)
                m=torch.where(torch.isinf(m),torch.full_like(m,0),m)
                if is_compare and m.shape[-1]>500:
                    filename_block=f'{file_pre}/{dataset}/{dataset}_{sampling_order}_inf.txt'
                    with open(filename_block, 'a') as f:
                        cos_sim=torch.sum(x*m)/torch.norm(x)/torch.norm(m)
                        f.write(f'{cos_sim}\n')
                        print(cos_sim)
                # print('forward:',torch.norm(m-x),torch.max(torch.abs(m-x)))
                return m
            elif y_norm>1e13:
                return x
            else:
                return x-x 
        else:
            return x

    @staticmethod
    def backward(ctx, x):
        # 在backward方法中，获取保存的input
        input, sampling_order= ctx.saved_tensors
        if sampling_order != 0:
            x_max=torch.max(torch.abs(x))
            y=x.transpose(-1,-2)/x_max
            yn=y/torch.unsqueeze(torch.norm(y,dim=-1),dim=-1)
            yn=torch.where(torch.isnan(yn),torch.full_like(yn,0),yn)
            # yn=torch.where(torch.isinf(yn),torch.full_like(yn,0),yn)
            y_norm=y.norm(2)
            # y=torch.squeeze(torch.flatten(x,0,x.dim()-1),0)
            sampling_times=int(sampling_order*1000)
            if y_norm>1e-8 and y_norm<1e13:
                # y_abs=y/y_norm*y/y_norm 

                # m=torch.distributions.multinomial.Multinomial(sampling_times,torch.abs(yn) ** 2).sample()/sampling_times
                m=torch.distributions.binomial.Binomial(sampling_times,torch.abs(yn) ** 2).sample()
                m=m/torch.unsqueeze(torch.sum(m,dim=-1),dim=-1)

                new_states = torch.cat(((yn+torch.sqrt(m))/torch.sqrt(torch.tensor(2.0)),(yn-torch.sqrt(m))/torch.sqrt(torch.tensor(2.0))),dim=-1)
                new_states=(new_states/torch.unsqueeze(torch.norm(new_states,dim=-1),dim=-1))**2
                # outcomes=torch.distributions.multinomial.Multinomial(sampling_times,new_states).sample()

                new_states=torch.where(torch.isnan(new_states) ,torch.full_like(new_states,0),new_states)
                new_states=torch.where(torch.isinf(new_states),torch.full_like(new_states,0),new_states)
                outcomes=torch.distributions.binomial.Binomial(sampling_times,new_states).sample()
                outcomes=outcomes/torch.unsqueeze(torch.sum(outcomes,dim=-1),dim=-1)
                n_0_i = outcomes[...,:y.shape[-1]]
                sigma_i = torch.where(n_0_i/m > 0.4 , torch.tensor(1.0), torch.tensor(-1.0))
                m = sigma_i * torch.sqrt(m)

                # m=m/m.norm(2)*y_norm 
                m1=torch.norm(y,dim=-1)/torch.norm(m,dim=-1)
                m1=m1.unsqueeze(dim=-1)
                m=m*m1*x_max
                m=m.transpose(-1,-2)
                m=torch.where(torch.isnan(m),torch.full_like(m,0),m)
                m=torch.where(torch.isinf(m),torch.full_like(m,0),m)
                # print('back:',torch.norm(m-x),torch.max(torch.abs(m-x)))
                return m,None 
            else:
                return x-x, None
        else: 
            return x, None

class LinearSampling(torch.nn.Linear):

    def __init__(self,in_features, out_features, bias=True,sampling_error=0.,sampling_mode=0,sampling_order=100):
        super(LinearSampling, self).__init__(in_features, out_features, bias)
        self.sampling_error=sampling_error
        self.sampling_mode=sampling_mode
        self.sampling_order=sampling_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LinearFunctionSampling.apply(x,self.weight,self.bias,self.sampling_error,self.sampling_mode,self.sampling_order)

class LinearFunctionSampling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight,bias,sampling_error,sampling_mode,sampling_order):
        # 保存反向传播所需的参数
        ctx.save_for_backward(input, weight, bias,Tensor([sampling_error]),Tensor([sampling_mode]),Tensor([sampling_order]))
        # output = input.mm(weight.t())
        # if bias is not None:
        #     output += bias.unsqueeze(0).expand_as(output)
        output=F.linear(input,weight,bias)
        if compute_success_prob_block:
            output1=F.linear(input,weight,bias=None)
            f1=torch.norm(input).cpu()
            f2=torch.norm(weight).cpu()
            f3=torch.norm(output1).cpu()
            if f3>1e-6:
                y_size=output1.size()
                y_dim=1
                for s in y_size:
                    y_dim=y_dim*s
                sqrt_p=(f3/f1/f2).numpy()
                filename=f'{file_pre}/{dataset}/{dataset}_{r_size}_block-encoding_forward.txt'
                with open(filename, 'a') as f:
                    f.write(f'{y_dim} {sqrt_p}\n') 
                # print(f'block:prob={inv_prob}-------{input.size()},{weight.size()},{output1.size()}')

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的参数
        input, weight,bias,sampling_error,sampling_mode,sampling_order = ctx.saved_tensors
        # 对梯度进行一些后处理
        sampling_engine=SamplingBlock(sampling_error[0],sampling_mode[0],sampling_order[0])
        grad_input = torch.matmul(grad_output,weight)
        grad_weight = torch.matmul(torch.transpose(grad_output, grad_output.dim()-2,grad_output.dim()-1),input).sum(0) #reduce batch
        grad_bias = None
        if bias is not None:
            sampling_engine.sampling_order=grad_output.shape[-2]
            grad_bias = sampling_engine(grad_output) 
            grad_bias=grad_output.sum([axis for axis in range(grad_output.dim()-1)]) #reduce batch
        # add sampling process
        sampling_engine.sampling_order=grad_weight.shape[-2]
        grad_weight_samping=sampling_engine(grad_weight)
        if compute_success_prob_block:
            f1=torch.norm(grad_output).cpu()
            f2=torch.norm(weight).cpu()
            f3=torch.norm(grad_input).cpu()
            # g1=torch.norm(grad_output)
            g2=torch.norm(input).cpu()
            g3=torch.norm(grad_weight).cpu()
            filename=f'{file_pre}/{dataset}/{dataset}_{r_size}_block-encoding_backpropagation.txt'
            if f3>1e-6:
                y_size=grad_input.size()
                y_dim=1
                for s in y_size:
                    y_dim=y_dim*s
                sqrt_p=(f3/f1/f2).numpy()
                with open(filename, 'a') as f:
                    f.write(f'{y_dim} {sqrt_p}\n') 
            if g3>1e-6:
                y_size=grad_input.size()
                y_dim=1
                for s in y_size:
                    y_dim=y_dim*s
                sqrt_p=(g3/f1/g2).numpy()
                with open(filename, 'a') as f:
                    f.write(f'{y_dim} {sqrt_p}\n') 
        return grad_input, grad_weight_samping, grad_bias,None,None,None


class MultiheadAttentionSampling(BaseModule):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        use_layer_scale (bool): Whether to use layer scale. Defaults to False.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sampling_error=0.,
                 sampling_mode=0,
                 sampling_order=100,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 use_layer_scale=False,
                 layer_scale_init_value=0.,
                 init_cfg=None):
        super(MultiheadAttentionSampling, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        if qk_scale is not None:
            self.scaled_dot_product_attention = partial(
                scaled_dot_product_attention_pyimpl,
                scale=self.head_dims**-0.5)
        else:
            self.scaled_dot_product_attention = scaled_dot_product_attention

        self.qkv = LinearSampling(self.input_dims, embed_dims * 3, bias=qkv_bias,sampling_error=sampling_error,sampling_mode=sampling_mode,sampling_order=sampling_order)
        self.attn_drop = attn_drop
        self.proj = LinearSampling(embed_dims, embed_dims, bias=proj_bias,sampling_error=sampling_error,sampling_mode=sampling_mode,sampling_order=sampling_order)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = build_dropout(dropout_layer)

        if use_layer_scale:
            warnings.warn('The `use_layer_scale` in `MultiheadAttention` will '
                          'be deprecated. Please use `layer_scale_init_value` '
                          'to control whether using layer scale or not.')

        if use_layer_scale or (layer_scale_init_value > 0):
            layer_scale_init_value = layer_scale_init_value or 1e-5
            self.gamma1 = LayerScale(
                embed_dims, layer_scale_init_value=layer_scale_init_value)
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_drop = self.attn_drop if self.training else 0.
        x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))
        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x



class PESampling(nn.Module):

    def __init__(self,poistion_parameters,sampling_error,sampling_mode,sampling_order):
        super(PESampling, self).__init__()
        self.sampling_error=sampling_error
        self.sammpling_mode=sampling_mode
        self.sampling_order=sampling_order
        self.pos_parameters=poistion_parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return PESamplingAutoGrad.apply(x,self.pos_parameters,self.sampling_error,self.sammpling_mode,self.sampling_order)


class PESamplingAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,position_parameters,sampling_error,sampling_mode,sampling_order):
        ctx.save_for_backward(x, position_parameters, Tensor([sampling_error]),Tensor([sampling_mode]),Tensor([sampling_order]))
        return x+position_parameters

    @staticmethod
    def backward(ctx, grad_output):
        # 在backward方法中，获取保存的input
        input,position_parameters, sampling_error,sampling_mode,sampling_order= ctx.saved_tensors
        sampling_engine=SamplingBlock(sampling_error[0],sampling_mode[0],sampling_order[0])
        grad_output=sampling_engine(grad_output)
        return grad_output,grad_output, None,None,None

        # if sampling_error != 0:
        #     y=torch.squeeze(torch.flatten(grad_output,0,grad_output.dim()-1),0)
        #     sampling_times=int(np.log2(y.shape[0])/sampling_error/sampling_error)
        #     y_norm=y.norm(2)
        #     if y_norm>1e-8 and y_norm<1e13:

        #         y_abs=y/y_norm*y/y_norm 
        #         m=torch.distributions.binomial.Binomial(sampling_times,y_abs).sample()
        #         m=torch.sqrt(m)
        #         m=torch.sign(y)*m 
        #         m=m/m.norm(2)*y_norm 
        #         m=torch.reshape(m,grad_output.shape)
        #         return grad_output,m,None
        #     else:
        #         return grad_output-grad_output,grad_output-grad_output, None
        # else: 
        #     return grad_output,grad_output, None


