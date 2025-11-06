import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


LRELU_SLOPE = 0.1



class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_parametrizations(l, tensor_name='weight')
        for l in self.convs2:
            remove_parametrizations(l, tensor_name='weight')


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_parametrizations(l, tensor_name='weight')


def pad_diff(x):
    return F.pad(F.pad(x, (0,0,-1,1), 'constant', 0) - x, (0,0,0,-1), 'constant', 0)

class SineGen(torch.nn.Module):
    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0
        ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        #f0_values 的尺寸是 batchsize, length, dim
        # 这里的dim 其实是 (f0, 2f0, 3f0 ....), 
        # 就是f0 乘以  range(1, self.harmonic_num + 2) 得来的
        # length 是时间维度

        rad_values = (f0_values / self.sampling_rate) % 1 #rad_values = kf_0/f_s

        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0 #当k=1，即f0上不需要加初始相位
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        #rad_values 是约等于公式 Δ𝜃 = k*f_0/f_s,  
        #但是在t0上加了初始相位 φ_k/(2𝜋)
        # rand_ini = (batchsize x dim), rad_values 在时间 t0 上加上 φ_k/(2𝜋)

        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (pad_diff(tmp_over_one)) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1)  * 2 * np.pi)
        #torch.cumsum(rad_values + cumsum_shift, dim=1) 
        # 等于 torch.cumsum(rad_values, dim=1)  % 1
        # 这里之所以用 cumsum_shift 数学上效果和 %1 一样，
        # 但实际计算中 cumsum_shift 的精度更高，
        # 因为减去cumsum_shift之后，求和的值更小，浮点数精度就更高
        # %1是因为 sin(1.2 * 2𝜋) = sin(0.2 * 2𝜋)
        #torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1)  * 2 * np.pi) 就是 离散公式s[n]
        return sines

    def forward(self, f0, upp=None):
        with torch.no_grad():
            fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
            sine_waves = self._f02sine(fn) * self.sine_amp

            uv = self._f02uv(f0)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            #noise= (1-uv[n]) * Noise(n) 的实现，工程上的实现
            sine_waves = sine_waves * uv + noise
            # 总公式 sinegen[n] 这里没有用谐波求和公式，各个分量没有直接求和
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp=None):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs.to(self.l_linear.weight.dtype)))
        #没有手工求 谐波求和公式 sum-s[n]，而是通过 Linear 层来学习组合权重。

        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class HiFiGen(torch.nn.Module):
    def __init__(self, 
        sampling_rate,
        inter_channels,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels
    ):
        super(HiFiGen, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            harmonic_num=8)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(inter_channels, upsample_initial_channel, 7, 1, padding=3))
        Resblock = ResBlock1 if resblock == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u +1 ) // 2)))
            if i + 1 < len(upsample_rates):  #
                stride_f0 = np.prod(upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(Resblock( ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        self.upp = np.prod(upsample_rates)


    def forward(self, x, f0, g=None):#x=6x192x20, f0=6x20, g=6x768x1
        # print(1,x.shape,f0.shape,f0[:, None].shape)
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  #f0=6x10240x1
            
        # print(2,f0.shape)
        har_source, noi_source, uv = self.m_source(f0, self.upp)#har_source=6x10240x1
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)#x=6x512x20
        x = x + self.cond(g)#x=6x512x20
        # print(124,x.shape,har_source.shape)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            # print(3,x.shape)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            # print(4,x_source.shape,har_source.shape,x.shape)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)#x=6x16x10240
        x = self.conv_post(x)#x=6x1x10240
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_parametrizations(l, tensor_name='weight')
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_parametrizations(self.conv_pre, tensor_name='weight')
        remove_parametrizations(self.conv_post, tensor_name='weight')
