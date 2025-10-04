import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from .hifi_gen import HiFiGen
from .encoder import Encoder
from .text_encoder import TextEncoder
from .f0_decoder import F0Decoder
from .flow import Flow
from .modules import commons


class SynthTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        ssl_dim,
        n_speakers,
        sampling_rate=44100,
        flow_share_parameter=False,
        n_flow_layer=4,
    ):

        super().__init__()
        self.segment_size = segment_size
        #self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.spk_emb = nn.Linear(192, gin_channels)

        self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )
        hps = {
            "sampling_rate": sampling_rate,
            "inter_channels": inter_channels,
            "resblock": resblock,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels
        }

        self.dec = HiFiGen(h=hps)

        self.enc_q = Encoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = Flow(
            inter_channels,
            hidden_channels,
            5,
            1,
            n_flow_layer,
            gin_channels=gin_channels,
            share_parameter=flow_share_parameter,
        )

        self.f0_decoder = F0Decoder(
            1,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            spk_channels=gin_channels,
        )
            
        self.emb_uv = nn.Embedding(2, hidden_channels)
        self.character_mix = False


    def forward(
        self, c, f0, uv, spec, g=None, c_lengths=None, spec_lengths=None
    ):# c=32x768x799, f0,uv=32x799, spec=32x1025x799, g=32x192,  c_lengths=32x1, spec_lengths=32x1
        g = self.spk_emb(g).unsqueeze(-1)
        # g=32x768x1
        # vol proj
       
        # vol=0
        # ssl prenet
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )#x_mask=32x1x799
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2)#self.pre(c)=32x192x799, self.emb_uv(uv.long()).transpose(1, 2)=32x192x799
        #x=32x768x799
        # f0 predict
        lf0 = (
            2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500
        )# lf0=32x1x799
        norm_lf0 = commons.normalize_f0(lf0, x_mask, uv)  #norm_lf0=32x1x799
        pred_lf0 = self.f0_decoder(
            x, norm_lf0, x_mask, spk_emb=g
        )  # pred_lf0=32x1x799
        # encoder
        z_ptemp, m_p, logs_p, _ = self.enc_p(
            x, x_mask, f0=commons.f0_to_coarse(f0)#commons.f0_to_coarse量化到1-255个整数上
        )  #z_ptemp, m_p,logs_p=32x192x799,
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        # z, m_q, logs_q=32x192x799, spec_mask=32x1x799
        # flow
        z_p = self.flow(z, spec_mask, g=g)  # z_p=32x192x799
        z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z, f0, spec_lengths, self.segment_size
        )
        # ids_slice=32x1, pitch_slice=32x20, z_slice=32x192x20
        # nsf decoder
        o = self.dec(z_slice, g=g, f0=pitch_slice)
        # o=31x1x10240
        return (
            o,
            ids_slice,
            spec_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            pred_lf0,
            norm_lf0,
            lf0,
        )

    @torch.no_grad()
    def infer(
        self,
        c,
        f0,
        uv,
        g=None,
        noice_scale=0.35,
        seed=52468,
        predict_f0=False
    ):

        if c.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        if self.character_mix and len(g) > 1:  # [N, S]  *  [S, B, 1, H]
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            g = g * self.speaker_map  # [N, S, B, 1, H]
            g = torch.sum(g, dim=1)  # [N, 1, B, 1, H]
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0)  # [B, H, N]
        else:
            if g.dim() == 1:
                g = g.unsqueeze(0)
            g = self.spk_emb(g).unsqueeze(-1)

        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )
        # vol proj


        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2)

        lf0 = 2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500
        norm_lf0 = commons.normalize_f0(lf0, x_mask, uv, random_scale=False)
        pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
        f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
            

        z_p, m_p, logs_p, c_mask = self.enc_p(
            x, x_mask, f0 = commons.f0_to_coarse(f0), noice_scale=noice_scale
        )
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=f0)
        return o, f0
