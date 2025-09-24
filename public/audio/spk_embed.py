
import os
import sys
import re
import pathlib
import numpy as np
import argparse
import torch
import torchaudio
from speakerlab.process.processor import FBank
from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2



class SpkEmbed:
    def __init__(self, device, sample_rate:int):
        assert sample_rate == 16000
        model_dir = "/code/data/models/speech_eres2netv2_sv_zh-cn_16k-common/"
        model = ERes2NetV2(
            feat_dim = 80,
            embedding_size = 192,
            baseWidth = 26,
            scale = 2,
            expansion = 2,
        )
        pretrained_state = torch.load(model_dir + "pretrained_eres2netv2.ckpt")
        model.load_state_dict(pretrained_state)
        model.to(device)
        model.eval()
        self.model = model
        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    def get_embedding(self, batch_wavs:torch.Tensor):
        batch_size = batch_wavs.shape[0]
        feats = []
        for i in range(batch_size):
            feats.append(self.feature_extractor(batch_wavs[i]))
        embedding = self.model(torch.stack(feats))
        return embedding

def test_spk_embed():
    from public.toolkit.io import read_json, save_json, read_jsonl, log
    from public.audio import datasets
    import torch.nn.functional as F
    sample_rate = 16000
    device = "cuda"
    datasets.config.update(
        sample_rate=sample_rate,
        max_val_duration=12, 
        train_duration=2.4,
        speaker_duration=6.0,
        hop_length=320,
        device=device,
    )
    data:dict = read_json("/code/playground/audio-data/train_data_singer-8.0.json")["singers"]
    singers = list(data.keys())
    files = [ data[singers[0]][0], data[singers[0]][1], data[singers[1]][1], data[singers[2]][1]]
    wav =  datasets.wav2tensor(
        [datasets.get_sample(file, False, True) for file in files], 
        ["wav", "start", "end"]
    ).unsqueeze(1)
    spke = SpkEmbed(device=device, sample_rate=sample_rate)
    embed1 = spke.get_embedding(wav)
    x1 = embed1.unsqueeze(1) 
    x2 = embed1.unsqueeze(0)

    similarity_matrix = F.cosine_similarity(x1, x2, dim=-1)

    print(similarity_matrix)
    print(embed1)

