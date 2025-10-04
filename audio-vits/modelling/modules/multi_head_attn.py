import torch
from torch import nn
from torch.nn import functional as F
from . import commons
import math


class MultiHeadAttn(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        window_size=None,
        heads_share=True,
        block_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):#x=32x192x799, c=32x192x799, attn_mask=32x1x799x799
        q = self.conv_q(x)#q=32x192x799
        k = self.conv_k(c)#k=32x192x799
        v = self.conv_v(c)#v=32x192x799

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))#b=32, d=192, t_s=799, t_t=799
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)#query=32x2x799x96
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)#key=32x2x799x96
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)#value=32x2x799x96

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))#scores=32x2x799x799
        if self.window_size is not None:#self.window_size=4
            assert (
                t_s == t_t
            ), "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)#self.emb_rel_k=1x9x96, key_relative_embeddings=1x1597x96
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )#rel_logits=32x2x799x1597 其中1597中只有中间的9列有数据
            scores_local = self._relative_position_to_absolute_position(rel_logits)#scores_local=32x2x799x799
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert (
                    t_s == t_t
                ), "Local attention is only available for self-attention."
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)#output=32x2x799x96
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)#relative_weights=32x2x799x1597
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s
            )#value_relative_embeddings=1x1597x96
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )#output=32x2x799x96
        output = (
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):#x=32x2x799x96, y=1x1597x96
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))#ret=32x2x799x1597
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):#relative_embeddings=1x9x96, length=799
        2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)#pad_length=794
        slice_start_position = max((self.window_size + 1) - length, 0)#slice_start_position=0
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,#relative_embeddings=1x9x96
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),#[0, 0, 794, 794, 0, 0] [pad_last_dim_left, pad_last_dim_right, pad_second_last_dim_left, pad_second_last_dim_right, ...] 就是dim0和dim2保持不变，在dim1的前后各自填充了 794x94 个0
            )#padded_relative_embeddings=1x1597x96  1597=794+794+9
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings#used_relative_embeddings=1x1597x96

    def _relative_position_to_absolute_position(self, x):#x=32x2x799x1597
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))#x=32x2x799x1598

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])#x_flat=32x2x1276802
        x_flat = F.pad(
            x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )#x_flat=32x2x1277600

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]#x_flat.view([batch, heads, length + 1, 2 * length - 1])=32x2x800x1597, 
        return x_final#x_final=32x2x799x799

    def _absolute_position_to_relative_position(self, x):#x=32x2x799x799
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(
            x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )#x=32x2x799x1597
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])#x_flat=32x2x1276003
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))#x_flat=32x2x1276802
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]#x_flat.view([batch, heads, length, 2 * length])=32x2x799x1598
        return x_final#x_final=32x2x799x1597

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)
