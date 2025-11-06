
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
from config import config


class SpkEmbed:
    def __init__(self, device, sample_rate:int):
        assert sample_rate == 16000
        model_dir = config["features_spk_embed"]
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
