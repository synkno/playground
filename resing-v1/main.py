

import os
import sys
sys.dont_write_bytecode = True
code_dir = os.path.dirname(__file__)
#sys.path.append(code_dir)
sys.path.append(os.path.join(code_dir, "3rd-libs"))


from dataset.preproc import preproc_folder, create_datasets, create_spk_emb

#preproc_folder("/data/custom-datasets/re-sing-44k/wavs/", "/data/custom-datasets/re-sing-44k/datasets/")
#create_datasets("/data/custom-datasets/re-sing-44k/datasets/")


from stem.train import train, config

import torch
    
if __name__ == "__main__":

    import argparse
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    
    world_size = 2
    if args.local_rank != -1:
       train(args.local_rank, world_size)
    elif args.debug:
        config["debug"] = True
        train(0, 1)
    else:
        for i in range(world_size):
            subprocess.Popen(["python", "main.py", "--local_rank", str(i)])