import os
import sys
sys.dont_write_bytecode = True
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(code_dir)
sys.path.append(os.path.join(code_dir, "3rd-libs"))


from training.data_preproc import data_prepoc
from training.trainer import train, Unit2Mel, Vocoder, torch, utils, log, np, config

from training.data_loader import create_loader


def infer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    vocoder = Vocoder(device=device)

    unit2mel = Unit2Mel(
        input_channel=768, 
        n_spk = 1,
        out_dims = vocoder.dimension,
        n_layers = 20,
        n_chans = 512,
        n_hidden = 256,
        timesteps = 1000,
        k_step_max = 0
    )
    utils.load_checkpoint("/code/playground/audio-data/diff/73800_model.pth", unit2mel)
    unit2mel = unit2mel.to(device=device, dtype=dtype)

    loader = create_loader(
        [
            "/code/data/custom-datasets/kuaou/split-8.0-singer-44k/fd203e1119c3338600afbe836401922e-1.wav",
            "/code/data/custom-datasets/kuaou/split-8.0-singer-44k/fb2542f2e93a7be59fec10a9a3edc3f5-0.wav",
            "/code/data/custom-datasets/kuaou/split-8.0-singer-44k/8be1f8047ad8d615d69153e77db79347-0.wav"
        ],
        num_workers=0, is_train=False, batch_size=1, shuffle=False
    )
    from public.audio.spk_embed import SpkEmbed
    import soundfile
    from public.audio.datasets import load_audio
    spk_embed = SpkEmbed(device, sample_rate=16000)
    wav = load_audio("/code/data/custom-datasets/audio-examples/zaoanun-16k-mono.wav", sampling_rate=16000)
    
    spk_vec:torch.Tensor = spk_embed.get_embedding(torch.tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)) #1x192

    out_dir = "../audio-data/diff/"
    def np2torch(data):
        for k in data.keys(): 
            if not isinstance(data[k], np.ndarray):continue
            data[k] = torch.tensor( data[k], device=device, dtype=dtype)

    for batch_idx, data in enumerate(loader):
        np2torch(data)
        mel = unit2mel(
            units   = data['units'], 
            f0      = data['f0'], 
            volume  = data['volume'], 
            spk_emb = spk_vec, 
            gt_spec=None if unit2mel.k_step_max == unit2mel.timesteps else data['mel'],
            infer=True, 
            k_step=unit2mel.k_step_max
        )
        signal = vocoder.infer(mel.to(torch.float32), data['f0'].to(torch.float32))
        audio = vocoder.infer(data["mel"].to(torch.float32), data['f0'].to(torch.float32))

        soundfile.write(f"{out_dir}{batch_idx}_rec.wav", signal.detach().cpu().float().numpy().reshape(-1), config["sampling_rate"])
        soundfile.write(f"{out_dir}{batch_idx}_raw.wav", audio.detach().cpu().float().numpy().reshape(-1), config["sampling_rate"])

if __name__ == "__main__":
    infer()
    exit()

    import argparse
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    
    world_size = 2
    if args.local_rank != -1:
        train(args.local_rank, world_size)
        pass
    elif args.debug:
        train(0, 1)
        pass
    else:
        for i in range(world_size):
            subprocess.Popen(["python", "main.py", "--local_rank", str(i)])

