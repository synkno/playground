
import torch


config = {
    "sampling_rate": 44100,
    "hop_length" : 512,
    "filter_length": 2048,
    "segment_size": 10240,


    "out_dir" : "../out-dir/tmp7/",
    "net_g" : "/code/open-repo/out-dir/tmp6/models/28600_G.pth",
    "net_d" : "/code/open-repo/out-dir/tmp6/models/28600_D.pth",

    "learning_rate" : 0.001,
    "epochs": 50,
    "dtype" : torch.float32,


    
    "lr_decay": 0.91,
    "betas": [  0.8,  0.99  ],
    "eps": 1e-09,
    "log_interval": 200,
    "eval_interval": 2000,
    "batch_size": 24,
    "loss_weights" : {
        "mel": 45,
        "kl": 1.0,
    },

    "data_dir" : "/data/custom-datasets/re-sing-44k/",
    
    "features_spk_embed" : "/data/models/speech_eres2netv2_sv_zh-cn_16k-common/",
    "features_hubert_vec" : "/data/models/hubert_base/hubert_base.pt",

    "debug" : False
}