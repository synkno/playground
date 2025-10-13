import torch
config = {
    "sampling_rate": 44100,
    "out_dir" : "../audio-data/temp6/",
    "epochs": 300,
    "filter_length": 2048,
    "hop_length" : 512,
    "dtype" : torch.float32,
    "lr_step_size" : 40,
    "train_duraton" : 4.0,
    "model_file" : "/code/playground/audio-data/temp5/models/49200_model.pth",
}