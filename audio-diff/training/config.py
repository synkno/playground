import torch
config = {
    "sampling_rate": 44100,
    "out_dir" : "../audio-data/temp7/",
    "epochs": 1000,
    "filter_length": 2048,
    "hop_length" : 512,
    "dtype" : torch.float32,
    "lr_step_size" : 150,
    "train_duraton" : 4.0,
    "model_file" : "/code/playground/audio-data/temp6/models/73800_model.pth",
}