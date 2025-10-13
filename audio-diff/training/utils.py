import json
import os

import torch
from public.toolkit.io import log




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
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except Exception:
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
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({"model": state_dict}, checkpoint_path)
