from collections import deque
from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class GaussianDiffusion(nn.Module):
    def __init__(self, 
                denoise_fn, 
                out_dims=128,
                timesteps=1000, 
                k_step=1000,
                max_beta=0.02,
                spec_min=-12, 
                spec_max=2):
        
        super().__init__()
        self.denoise_fn = denoise_fn
        self.out_dims = out_dims
        betas =  np.linspace(1e-4, max_beta, timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.k_step = k_step if k_step>0 and k_step<timesteps else timesteps

        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.spec_min = spec_min
        self.spec_max = spec_max
   

    @torch.no_grad()
    def p_sample(self, x_t, t, cond):
        b, *_, device = *x_t.shape, x_t.device

        noise_pred = self.denoise_fn(x_t, t, cond=cond)
        #预测出的噪声

        x_0 = (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise_pred
        )#这里知道x_t, 反求x_0, 
        #x0 = 1/sqrt(α̅t) * xt + sqrt(1-α̅t)/sqrt(α̅t) * noise

        x_0.clamp_(-1., 1.)#强制把数字限制在 [-1, 1]之间


        model_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )#求的是x_{t-1}分布的均值μ(x_t, x_0) 
        #μ(x_t, x_0) = sqrt(ᾱ_{t-1})*β_t / (1-ᾱ_t) * x_0 + sqrt(α_t)*(1-ᾱ_{t-1}) / (1-ᾱ_t) * x_t

        model_variance = extract(self.posterior_variance, t, x_t.shape)
        #求的是x_{t-1}分布的方差 σ^2 = (1-ᾱ_{t-1})*β_t / (1-ᾱ_t)
        model_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        #求的是x_{t-1}分布的方差的log log(σ^2)

        noise = torch.randn(x_t.shape, device=device, dtype=x_t.dtype)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1))).to(x_t.dtype)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        #nonzero_mask 作用是 当t==0时，不要加noise, 否则加noise
        #(0.5 * model_log_variance).exp() = exp(1/2*log(var)) = exp(log( sqrt(var) )) = sqrt(var)


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )#扩散模型的正向公式加noise, xt = sqrt(α̅t) * x0 + sqrt(1-α̅t) * noise

    def p_losses(self, x_start, t, cond):#x_start=8x1x128x172, t=8x, cond=8x256x172, 
        noise = torch.randn_like(x_start)#noise=8x1x128x172
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)#x_noisy=8x1x128x172
        # 1. 前向扩散：往真实 mel (x_start) 里加噪声

        x_recon = self.denoise_fn(x_noisy, t, cond)
        # 2. 模型预测：输入 (x_noisy, t, cond)，输出预测的噪声, 其中t是随机的扩散步数

        loss = F.mse_loss(noise, x_recon)
        #3. 损失函数：预测噪声 vs 真实噪声

        return loss

    def forward(self, 
                condition, 
                gt_spec=None, 
                infer=True, 
                k_step=300):
        
        cond = condition.transpose(1, 2)
        b, device = condition.shape[0], condition.device

        if not infer:
            spec = self.norm_spec(gt_spec)#spec, gt_spec=8x172x128 gt_spec是真实音频mel
            t = torch.randint(0, self.k_step, (b,), device=device).long()#t=8
            norm_spec = spec.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            return self.p_losses(norm_spec, t, cond=cond)
        
        shape = (cond.shape[0], 1, self.out_dims, cond.shape[2])#(1, 1, 128, 768)
        if gt_spec is None:
            t = self.k_step
            x = torch.randn(shape, device=device, dtype=condition.dtype)
        else:
            t = k_step
            norm_spec = self.norm_spec(gt_spec)
            norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]
            x = self.q_sample(x_start=norm_spec, t=torch.tensor([t - 1], device=device).long())
        #x=1x1x128x768
        for i in reversed(range(0, t)):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
                
        x = x.squeeze(1).transpose(1, 2)  # [B, T, M]
        return self.denorm_spec(x)
            

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

        

        



