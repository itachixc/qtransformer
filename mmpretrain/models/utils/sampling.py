
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
dataset='oxford_iii_pets'
file_pre='prob_results_0123'

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

class SamplingBlock(nn.Module):
    def __init__(self,sampling_error=0.):
        super(SamplingBlock, self).__init__()
        self.sampling_error=sampling_error
    def forward(self,x):
        chebyshev_dec_test(x,400)
        return SamplingBlockAutoGrad.apply(x,self.sampling_error)
        

class SamplingBlockAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,sampling_error):
        # 在forward方法中，保存input，供backward使用
        ctx.save_for_backward(x,Tensor([sampling_error]))
        if sampling_error != 0:
            y=torch.squeeze(torch.flatten(x,0,x.dim()-1),0)
            if compute_success_prob:
                y_max=torch.norm(y,float('inf')).cpu()
                y_f=torch.norm(y).cpu()
                y_size=y.size()[0]
                if y_max>1e-6:
                    sqrt_p=(y_f/y_max/np.sqrt(y_size)).numpy()
                    filename=f'{file_pre}/{dataset}/{dataset}_{r_size}_qdac_forward.txt'
                    with open(filename, 'a') as f:
                        f.write(f'{y_size} {sqrt_p}\n')
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

class LinearSampling(torch.nn.Linear):

    def __init__(self,in_features, out_features, bias=True,sampling_error=0.):
        super(LinearSampling, self).__init__(in_features, out_features, bias)
        self.sampling_error=sampling_error


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return LinearFunctionSampling.apply(x,self.weight,self.bias,self.sampling_error)

class LinearFunctionSampling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight,bias=None,sampling_error=0.):
        # 保存反向传播所需的参数
        ctx.save_for_backward(input, weight, bias,Tensor([sampling_error]))
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
        input, weight,bias,sampling_error = ctx.saved_tensors
        # 对梯度进行一些后处理
        sampling_engine=SamplingBlock(sampling_error)
        grad_input = torch.matmul(grad_output,weight)
        grad_weight = torch.matmul(torch.transpose(grad_output, grad_output.dim()-2,grad_output.dim()-1),input)
        grad_bias = None
        if bias is not None:
            grad_bias = sampling_engine(grad_output.sum(grad_output.dim()-2))
        # add sampling process
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
            
        
        return grad_input, grad_weight_samping, grad_bias,None


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

        self.qkv = LinearSampling(self.input_dims, embed_dims * 3, bias=qkv_bias,sampling_error=sampling_error)
        self.attn_drop = attn_drop
        self.proj = LinearSampling(embed_dims, embed_dims, bias=proj_bias,sampling_error=sampling_error)
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

    def __init__(self,poistion_parameters,sampling_error):
        super(PESampling, self).__init__()
        self.sampling_error=sampling_error
        self.pos_parameters=poistion_parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return PESamplingAutoGrad.apply(x,self.pos_parameters,self.sampling_error)


class PESamplingAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,position_parameters,sampling_error):
        ctx.save_for_backward(x, position_parameters, Tensor([sampling_error]))
        return x+position_parameters

    @staticmethod
    def backward(ctx, grad_output):
        # 在backward方法中，获取保存的input
        input,position_parameters, sampling_error= ctx.saved_tensors
        if sampling_error != 0:
            y=torch.squeeze(torch.flatten(grad_output,0,grad_output.dim()-1),0)
            sampling_times=int(np.log2(y.shape[0])/sampling_error/sampling_error)
            y_norm=y.norm(2)
            if y_norm>1e-8 and y_norm<1e13:

                y_abs=y/y_norm*y/y_norm 
                m=torch.distributions.binomial.Binomial(sampling_times,y_abs).sample()
                m=torch.sqrt(m)
                m=torch.sign(y)*m 
                m=m/m.norm(2)*y_norm 
                m=torch.reshape(m,grad_output.shape)
                return grad_output,m,None
            else:
                return grad_output-grad_output,grad_output-grad_output, None
        else: 
            return grad_output,grad_output, None


class ChebyshevDecomposition(nn.Module):
    def __init__(self,order):
        super(ChebyshevDecomposition, self).__init__()
        self.order=order
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x


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