
import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F




from modules.hifi_gen import HiFiGen
from modules.posterior_encoder import PosteriorEncoder
from modules.prior_encoder import PriorEncoder
from modules.flow import Flow
from modules import commons
from dataset.features.mel_spec import MelSpectrogram

def slice_segments(x:torch.Tensor, ids_str, segment_size=4):
    B, C, T = x.shape
    idx = torch.arange(segment_size, device=x.device).unsqueeze(0) + ids_str.unsqueeze(1)  # shape: (B, segment_size)
    idx = idx.clamp(0, T - 1) 
    idx = idx.unsqueeze(1).expand(-1, C, -1) 
    return torch.gather(x, dim=2, index=idx)


class ResingVits(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
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
        melspec:MelSpectrogram ,
        sampling_rate=44100,
        flow_share_parameter=False,
        n_flow_layer=4,
        
    ):

        super().__init__()
        #self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.spk_emb = nn.Linear(192, gin_channels)
        self.spk_prior = nn.Linear(192, hidden_channels)

        self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

        self.prior_enc = PriorEncoder(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )
        self.hifi_gen = HiFiGen(
            sampling_rate = sampling_rate,
            inter_channels = inter_channels,
            resblock = resblock,
            resblock_kernel_sizes = resblock_kernel_sizes,
            resblock_dilation_sizes = resblock_dilation_sizes,
            upsample_rates = upsample_rates,
            upsample_initial_channel = upsample_initial_channel,
            upsample_kernel_sizes = upsample_kernel_sizes,
            gin_channels = gin_channels
        )
        self.poster_enc = PosteriorEncoder(
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
            gin_channels=0,
            share_parameter=flow_share_parameter,
        )

        self.emb_uv = nn.Embedding(2, hidden_channels)
        self.melspec = melspec
       


    def forward(
        self, 
        content:torch.Tensor, 
        f0:torch.Tensor, uv:torch.Tensor, 
        spec:torch.Tensor, spk_emb:torch.Tensor, 
        lengths:torch.Tensor, wav:torch.Tensor,
        segment_size:int, hop_length:int,
    ):# c=32x768x799, f0,uv=32x799, spec=32x1025x799, g=32x192,  c_lengths=32x1, spec_lengths=32x1
        
        spk_emb = F.normalize(spk_emb, p=2, dim=1) 
        spk_prior = self.spk_prior(spk_emb).unsqueeze(-1)

        spk_emb = self.spk_emb(spk_emb).unsqueeze(-1)
        #self.spk_emb[nn.Linear(192, gin_channels)]
        #32x192 -> 32x768x1

        x_mask = torch.unsqueeze(
            commons.sequence_mask(lengths, content.size(2)), 1
        ).to( content.dtype  )#x_mask=32x1x799

        x = self.pre(content) * x_mask + self.emb_uv(uv.long()).transpose(1, 2)
        #self.pre[nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)]
        #self.emb_uv[nn.Embedding(2, hidden_channels)]
        #32x768x799 -> 32x192x799
        
        z_prior, m_prior, logs_prior, _ = self.prior_enc(
            x, x_mask, f0=commons.f0_to_coarse(f0), spk_emb=spk_prior
            #commons.f0_to_coarse量化到1-255个整数上
        )


        

        z_post, m_poster, logs_poster, spec_mask = self.poster_enc(
            spec, lengths,
        )

        z_poster = self.flow(z_post, spec_mask, )  # z_p=32x192x799


        slice_size = segment_size//hop_length

        z_slice, f0_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z_post, f0, lengths, slice_size
        )
        # ids_slice=32x1, pitch_slice=32x20, z_slice=32x192x20
        # nsf decoder
        wav_hat = self.hifi_gen(z_slice, g=spk_emb, f0=f0_slice)
        # o=31x1x10240

        mel = self.melspec.get_mel_from_spec(spec)
        mel = slice_segments(mel, ids_slice, slice_size)
        mel_hat = self.melspec.get_mel(wav_hat.squeeze(1) )
        wav = slice_segments(wav, ids_slice * hop_length, segment_size)  # slice


        return (
            spec_mask,
            mel, mel_hat,
            wav, wav_hat,
            z_poster, logs_poster, m_prior, logs_prior
        )
    
    @torch.no_grad()
    def infer(
        self,
        content:torch.Tensor,
        f0:torch.Tensor,
        uv:torch.Tensor,
        spk_emb:torch.Tensor,
        noice_scale=0.35,
        seed=52468
    ):

        if content.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)
        spk_emb = F.normalize(spk_emb, p=2, dim=1) 
        
        spk_prior = self.spk_prior(spk_emb).unsqueeze(-1)
        spk_emb = self.spk_emb(spk_emb).unsqueeze(-1)
        c_lengths = (torch.ones(content.size(0)) * content.size(-1)).to(content.device)
        x_mask = torch.unsqueeze(
            commons.sequence_mask(c_lengths, content.size(2)), 1
        ).to( content.dtype  )
        # vol proj
        x = self.pre(content) * x_mask + self.emb_uv(uv.long()).transpose(1, 2)

        z_p, m_p, logs_p, c_mask = self.prior_enc(
            x, x_mask, f0 = commons.f0_to_coarse(f0), noice_scale=noice_scale, spk_emb=spk_prior
        )

       
        z = self.flow(z_p, c_mask, reverse=True)
        wav_hat = self.hifi_gen(z * c_mask, g=spk_emb, f0=f0)
        return wav_hat
