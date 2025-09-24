
import torch
config = {
    "epochs" : 200,
    "out_dir" : "temp-20",
    "hop_length" : 160, #
    "max_datasets_len" : 0,
    "max_lr" : 1e-3,
    "latent_dim" : 512,
    "resblock_kernel_sizes" : [3, 5, 7, 9, 11, 13],
    "resblock_dilation_sizes" : [
        (1, 3, 5),
        (1, 4, 6),
        (1, 6, 9),
        (1, 2, 4),
        (1, 5, 7),
        (1, 8, 12)
    ],
    "dtype" : torch.bfloat16,
    "decoder" : "v0",
    "use_disc" : False,
    "model_file" : "/code/playground/audio-data/temp-19/temp-training/199-37000/model.torch"
}
# 最终效果最好的一版的配置, 训练了400epoch