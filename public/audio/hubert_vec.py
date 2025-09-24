

import torch
from fairseq import checkpoint_utils



class HubertVec:
    def __init__(self, device):
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
          ["/code/data/models/hubert_base/hubert_base.pt"],
          suffix="",
        )
        self.model = models[0].to(device)
        self.model.eval()
    
    def get_batch_embedding(self, batch_wavs:torch.Tensor):
        wavs = []
        for wav in batch_wavs:
            wavs.append(self.get_embedding(wav))
        return torch.cat(wavs, dim=0)

    def get_embedding(self, wav:torch.Tensor):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
          "source": feats.to(wav.device),
          "padding_mask": padding_mask.to(wav.device),
          "output_layer": 12,  # layer 12
        }
        logits = self.model.extract_features(**inputs)
        return logits[0].transpose(1, 2)
