from typing import Any, Dict, Literal
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class VectorQuantize(nn.Module):
    def __init__(self, 
        input_dim:int,
        codebook_size:int,
        codebook_dim:int,
        decay:float = 0.99,
        eps:float = 1e-5,
        reset_threshold:int = 2,
        mode:Literal["ste", "ema", "softmax"] = "softmax"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps
        self.reset_threshold = reset_threshold

        self.inproj = nn.Conv1d(input_dim, self.codebook_dim, kernel_size=1)
        self.outproj =nn.Conv1d(self.codebook_dim, input_dim, kernel_size=1)

        embed = torch.randn(codebook_size, codebook_dim)
        #self.register_buffer("embedding", embed)
        self.embedding = nn.Parameter(embed, requires_grad=True)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("codebook", embed.clone())
        self.softmax_temp = 1.0
        self.mode = mode
        self.ema_strategy = "warmup"
    

    def __get_distances(self, z_d:torch.Tensor):
        embedding:torch.Tensor = self.embedding
        z_norm2 = (z_d ** 2).sum(dim=1, keepdim=True)          # [N, 1]
        e_norm2 = (embedding ** 2).sum(dim=1).t()   # [1, K]
        distances = z_norm2 - 2 * (z_d @ embedding.t()) + e_norm2
        return distances
    def box_result(self, ** kargs):
        result = { "active_num": (self.cluster_size > self.reset_threshold).sum().float() }
        indices = kargs.pop("indices", None)
        if indices is not None:
            encodings = F.one_hot(indices, self.codebook_size).type(torch.float)
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            cluster_size_batch = encodings.sum(dim = 0)
            self.cluster_size.mul_(self.decay).add_(cluster_size_batch, alpha=1-self.decay)
            result["perplexity"] = perplexity
        return kargs | result
    
    def __forward_ema(self, z:torch.Tensor):
        z_e:torch.Tensor = self.inproj(z) #z_e: [B, D, T]
        z_d = rearrange(z_e, "b d t -> (b t) d")
        distances = self.__get_distances(z_d)

        indices = torch.argmin(distances, dim=1)   
        z_q = F.embedding(indices, self.embedding)
        z_q = rearrange(z_q, "(b t) d -> b d t", b=z_e.size(0))
        
        with torch.no_grad():
            encodings = F.one_hot(indices, self.codebook_size).type(z_d.dtype)

            #encodings: one_hot [BxT, CodeSize],  flat_inputs:[BxT, Dim]
            cluster_size_batch = encodings.sum(dim = 0)
            embed_sum_bacth = encodings.t() @ z_d
            #embed_sum_bacth: [CodeSize, Dim]
            """
            设: CodeSize = 3, BxT = 1, Dim = 2, 
                ET = encodings.t(), F = flat_inputs
            ET [ 0,     @  F [ 1, 1 ]
                1,           
                0 ]
            ET 行是codebook索引，列是时间步，每个元素是是否该时间步用了该索引。
            F  行是时间步，列是codebook的dim
            ET @ F 表示时间步用到的codebook的索引和codebook的dim 加权求和 
            """
            cluster_size:torch.Tensor = self.cluster_size
            codebook:torch.Tensor  = self.codebook

            cluster_size.mul_(self.decay).add_(cluster_size_batch, alpha=1-self.decay)
            codebook.mul_(self.decay).add_(embed_sum_bacth, alpha=1 - self.decay)

            n = cluster_size.sum()
            cluster_size = (
                (cluster_size + self.eps) / ( n + self.codebook_size * self.eps) * n
            )
            """
            n = sum(cluster_size)
            (cluster_size + self.eps) / ( n + self.codebook_size * self.eps) 有点归一化的感觉
            最后乘以n再恢复回去
            """
            """
            dead_codes = cluster_size < self.reset_threshold
            if dead_codes.any():
                rand_indices = torch.randint(0, flat_inputs.size(0), (dead_codes.sum(), ), device=flat_inputs.device)
                codebook[dead_codes] = flat_inputs[rand_indices]
            """
            self.embedding.copy_(codebook/cluster_size.unsqueeze(1))

            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        vq_loss = F.mse_loss(z_e, z_q.detach())
        z_q = z_e if self.ema_strategy == "warmup" else (z_e + (z_q - z_e).detach())
        z_q = self.outproj(z_q)
        
        return self.box_result(z_q=z_q, perplexity=perplexity, vq_loss=vq_loss )
    
    def __forward_ste(self, z:torch.Tensor):
        z_e:torch.Tensor = self.inproj(z) #z_e: [B, D, T]
        z_d = rearrange(z_e, "b d t -> (b t) d")
        distances = self.__get_distances(z_d)

        indices = torch.argmin(distances, dim=1)   
        z_q = F.embedding(indices, self.embedding)
        z_q = rearrange(z_q, "(b t) d -> b d t", b=z_e.size(0))

        commit_loss = ( F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2]) )
        codebook_loss = ( F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2]))
        z_q = z_e + (z_q - z_e).detach()
        z_q = self.outproj(z_q)
        return self.box_result(z_q=z_q, commit_loss=commit_loss, codebook_loss=codebook_loss, indices=indices)
    
    def __forward_softmax(self, z:torch.Tensor):
        z_e:torch.Tensor = self.inproj(z) #z_e: [B, D, T]
        z_d = rearrange(z_e, "b d t -> (b t) d")
        distances = self.__get_distances(z_d)

        logits = -distances
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            logits = logits + gumbel_noise
        y_soft = F.softmax(logits / self.softmax_temp, dim=-1)
        z_q = torch.matmul(y_soft, self.embedding)

        avg_probs = y_soft.mean(dim=0)  
        target = torch.full_like(y_soft, 1.0 / avg_probs.size(0))
        kl_loss = F.kl_div((avg_probs + 1e-10).log(), target, reduction='sum')

        indices = torch.argmax(y_soft, dim=1)

        z_q = rearrange(z_q, "(b t) d -> b d t", b=z_e.size(0))
        vq_loss = F.mse_loss(z_e, z_q.detach())

        z_q = self.outproj(z_q)

        return self.box_result(z_q=z_q, vq_loss=vq_loss, kl_loss=kl_loss, indices=indices)

    

    def forward(self, z:torch.Tensor)->Dict[str, Any]:
        if self.mode == "softmax":
            return self.__forward_softmax(z)
        if self.mode == "ema":
            return self.__forward_ema(z)
        return self.__forward_ste(z)