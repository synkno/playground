import torch
config = {
    "sampling_rate": 44100,
    "out_dir" : "../audio-data/temp3/",
    "net_g" : "/code/playground/audio-data/vits/49200_G.pth",
    "net_d" : "/code/playground/audio-data/vits/49200_D.pth",
    "learning_rate" : 0.00001,
    "epochs": 200,
    "filter_length": 2048,
    "hop_length" : 512,
    "dtype" : torch.bfloat16,

    "spk_map" : {
        "speaker5": 0,    "speaker18": 1,    "speaker40": 2,    "speaker29": 3,    "speaker21": 4,    "speaker2": 5,    "speaker1": 6,    
        "speaker41": 7,    "speaker24": 8,    "speaker14": 9,    "speaker43": 10,    "speaker3": 11,    "speaker19": 12,    "speaker38": 13,    
        "speaker46": 14,    "speaker39": 15,    "speaker15": 16,    "speaker11": 17,    "speaker30": 18,    "speaker12": 19,    
        "speaker31": 20,    "speaker27": 21,    "speaker35": 22,    "speaker32": 23,    "speaker20": 24,    "speaker28": 25,   
        "speaker33": 26,    "speaker26": 27,    "speaker9": 28,    "speaker44": 29,    "speaker25": 30,    "speaker42": 31,    
        "speaker17": 32,    "speaker22": 33,    "speaker10": 34,    "speaker6": 35,    "speaker36": 36,    "speaker34": 37,    
        "speaker13": 38,    "speaker8": 39,    "speaker4": 40,    "speaker37": 41,    "speaker7": 42,    "speaker16": 43,    
        "speaker45": 44,    "speaker23": 45,    "speaker0": 46
    },
    "n_speakers"  : 47,
}