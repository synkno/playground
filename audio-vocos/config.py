import torch
config = {
    "out_dir" : "../audio-data/temp2/",
    "epochs" : 80,
    "learning_rate" : 0.001,
    "lr_decay": 0.95,

    "sampling_rate": 24000,
    "n_mels" : 128,
    "n_fft" : 2048, 
    "hop_length" : 256, 
    "padding": "center",

    "net_g" : None,
    "net_d" : None, #"/code/playground/audio-data/temp1/models/19050_G.pth",

    "model":{
        "dim" : 768,#512,
        "intermediate_dim" : 2048, #1536,
        "num_layers" : 12, #8,
    },




    "dtype" : torch.float32,
    
    "betas": [  0.8,  0.99  ],
    "eps": 1e-09,

    "loss_weights" : {
        "loss_mr" : 0.1,
        "loss_gen_mr" : 0.1,
        "loss_fm_mr" : 0.1,
        "mel_loss" : 45,
    },

    "log_interval" : 100,
    "eval_interval" : 1000,

    "train_duration" : 2.4,
    "eval_duration" : 0,
    "batch_size" : 24,
    "data_dir" : "/data/custom-datasets/vocos-24k/",

    "debug" : False



}