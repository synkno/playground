import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

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
