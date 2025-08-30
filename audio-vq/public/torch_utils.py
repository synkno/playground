import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.optim.lr_scheduler import LambdaLR

def init_weights(m, target_types):
    if isinstance(m, target_types):
        nn.init.trunc_normal_(m.weight, std=0.02)
        #使用 截断正态分布（truncated normal distribution） 初始化卷积核的权重。
        #std=0.02 表示标准差为 0.02，均值默认为 0。
        #截断正态分布意味着：如果生成的值太远离均值（通常超过两倍标准差），就会被重新采样，使权重分布更集中、更稳定。
        nn.init.constant_(m.bias, 0)
        
def clip_optimizer_grad(optimizer: optim.Optimizer,  clip_val: float):
    #防止梯度爆炸，提升训练稳定性。
    torch.nn.utils.clip_grad_norm_([ps for g in optimizer.param_groups for ps in g["params"]], clip_val)

def set_requires_grad(model: nn.Module, flag: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad = flag 
    

__stored_params = {}
def __print_json(data, file:str = None):
    if file is None:
        print(json.dumps(data, indent=4))
        return
    with open(file,'a+',encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n\n")

def start_update_params(model:nn.Module):
    global __stored_params
    __stored_params = {name: param.clone() for name, param in model.named_parameters()}

def stop_update_params(model:nn.Module, file:str = None):
    global __stored_params
    names = [] 
    for name, param in model.named_parameters():
        if not torch.equal(__stored_params[name], param):
            names.append(name)
    __print_json(names, file)

def list_grad_params(model:nn.Module, file:str = None):
    names = [] 
    for name, param in model.named_parameters():
        if param.grad is not None:
            names.append(name)
    __print_json(names, file)

#from public.torch_utils import list_grad_params, start_update_params, stop_update_params


class WarmupLR(LambdaLR):
    def __init__(
        self, 
        optimizer:optim.Optimizer, 
        warmup_step:int = 0,
        #预热阶段的步数
        down_step:float = 5e4, 
        #从最大学习率下降到最小学习率的步数
        max_lr:float = 1e-4,
        min_lr:float = 1e-5, 
        last_epoch = -1
    ):
        self.warmup_step = warmup_step
        self.down_step = down_step
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.alpha = (self.max_lr - 1e-5)/self.warmup_step ** 2
        super().__init__(optimizer, self.lr_lambda, last_epoch)
    
    def lr_lambda(self, step):
        init_lr = 1e-5
        s1, s2 = self.warmup_step, self.warmup_step + self.down_step
        if step < s1:
            return init_lr + self.alpha * step ** 2 #从1e-5增长到max_lr, 按二次函数增长。
        elif s1 <= step < s2:
            return (self.max_lr - self.min_lr) / (s1 - s2) * step + (self.min_lr * s1 - self.max_lr * s2) / (s1 - s2)
            #线性下降, 等于 self.min_lr + (1 - (step-s1)/ ( s2 - s1) ) * (self.max_lr - self.min_lr) 
        else:
            return self.min_lr 
            #稳定期 

