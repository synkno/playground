import torch
import os
from public.io import log
import re

def load_checkpoint(file, model):
    if not file or not os.path.exists(file):
        return False

    checkpoint_dict = torch.load(file, map_location="cpu")
    saved_state_dict = checkpoint_dict["model"]
    model = model.to(list(saved_state_dict.values())[0].dtype)
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            #old_k = str(k).replace("prior_enc.attn_enc", "prior_enc.enc_").replace("poster_enc.wav_net", "poster_enc.enc")
            #old_k = re.sub(r"(flow\.flows\.\d+)\.wav_net", r"\1.enc", old_k)
            old_k = k
            new_state_dict[k] = saved_state_dict[old_k]
            assert saved_state_dict[old_k].shape == v.shape, (
                saved_state_dict[old_k].shape,
                v.shape,
            )
        except Exception:
            if "enc_q" not in k or "emb_g" not in k:
                log(
                    "%s is not in the checkpoint,please check your checkpoint.If you're using pretrain model,just ignore this warning."
                    % k
                )
                new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return True


def save_checkpoint(model, checkpoint_path):
    if not model: return
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({"model": state_dict}, checkpoint_path)

