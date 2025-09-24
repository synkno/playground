
from public.toolkit.object import Num
import os
import random
from typing import Dict, Any, Literal
import numpy as np
import torch
from operator import itemgetter
from public.toolkit.io import read_json, read_str, save_json, log, read_jsonl
import shutil
from pydub import AudioSegment
import re
import soundfile
import soxr
from pydub.silence import detect_silence


def split_audio_by_lrc(music_dir:str, out_dir:str, is_recreate:bool = False, separated_dir:str = None, min_duration:float = 0):
    if os.path.exists(out_dir) and not is_recreate:
        return
        
    data = read_json(os.path.join(music_dir, "data.json"))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    skip_silent = separated_dir is not None
    
    for pi, item in enumerate(data):
        lrc = read_str(os.path.join(music_dir, f"{item['id']}.lrc")).splitlines()
        if separated_dir is not None:
            singing_file = os.path.join(separated_dir, f"{item['id']}_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav")
        else:
            singing_file = os.path.join(music_dir, f"{item['id']}.wav")
            
        if not os.path.exists(singing_file):
            log(f"{singing_file} not found!")
            continue
        segs = []
        for line in lrc:
            match = re.search(r'\[(\d+),(\d+)\]', line)
            if not match: continue
            timestart = int(match.group(1))
            timeend = timestart + int(match.group(2))
            text = re.sub(r'<[^>]*>', '', line)
            text = text[text.find("]") + 1 : ].strip()
            segs.append({"start" : timestart, "text" : text, "end" : timeend})

        audio = AudioSegment.from_wav(singing_file)
        new_segs = []
        index = 0
        while index < len(segs):
            duration = 0
            texts = []
            times = []
            while index < len(segs):
                seg = segs[index]
                index += 1
                start,end = seg["start"], seg["end"]

                if skip_silent:
                    segment = audio[start:end]
                    if segment.dBFS < -40.0:
                        continue

                    if next((it for it in ["词：", "曲：", "唱：", "人：", "和声：","师：", "室："] if it in  seg["text"]), None) is not None:
                        continue

                    silent_ranges = detect_silence(segment, min_silence_len=200, silence_thresh=-40.0)
                    silent_duration = sum(end - start for start, end in silent_ranges)
                    if silent_duration/(end - start) > 0.5:
                        continue
                

                times.append([start, end])
                texts.append(seg["text"])
                duration += end - start
                if duration > min_duration: break
            if duration < min_duration: continue

            out_file = f"{out_dir}{item['id']}-{len(new_segs)}.wav"

            
            
            if skip_silent:
                new_times = []
                for start, end in times:
                    prev_end = new_times[-1][1] if new_times else 0
                    if prev_end > 0 and (start - prev_end) < 600:
                        new_times[-1][1] = end
                        continue
                    new_times.append([start, end])
                times = new_times
                duration = 0
                for start, end in times: duration +=  end - start

                seg_start = times[0][0]
                seg_end = seg_start + duration
                combined = AudioSegment.empty()
                for start, end in times:
                    combined += audio[start:end]
            else:
                seg_start = times[0][0]
                seg_end = times[-1][1]
                combined = audio[seg_start:seg_end]
            combined.export(out_file, format="wav")

            new_segs.append({
                "index" : len(new_segs), 
                "start" : seg_start, 
                "end" : seg_end,
                "times" : times,  
                "text" : " ".join(texts)
            })
        solo_sing = {
            "id" : item['id'],
            "name" : item["name"],
            "segs" : new_segs
        }
        save_json(f"{out_dir}{item['id']}.json", solo_sing)
        log(f"{(pi + 1)}/{len(data)} processed")
    log("compeleted")

def audio_volume_normalize(audio:np.ndarray, coeff:float = 0.2)->np.ndarray:
    temp = np.sort(np.abs(audio))
    if temp[-1] < 0.1:
        scaling_factor = max(temp[-1], 1e-3)
        audio = audio/scaling_factor * 0.1
        #如果音频的最大值都小于0.1，那么先把它放大到0.1

    temp = temp[temp > 0.01]
    L = temp.shape[0]
    #去掉小于0.01的值，要是样本数量小于10个，就直接返回
    if L <= 10:
        return audio
    
    volume = np.mean(temp[int(0.9 * L) : int(0.99 * L)])
    audio = audio * np.clip(coeff/volume, a_min=0.1, a_max=19)
    #volume取09-0.99的音量，用coeff/volume 缩放到 coeff 附近
    #coeff=0.2 标准语音或背景音，coeff=0.5 偏响亮但安全， 
    #coeff=0.05 非常轻柔的背景音. coeff=0.8 接近最大值，需谨慎使用.
    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio/max_value
        #防止超过1 音量失真。
    return audio

def load_audio(file:str, sampling_rate:int = None, volume_normalize: bool = False)->np.ndarray:
    audio, sr = soundfile.read(file)
    if audio.ndim == 2 and audio.shape[1] == 2:
        audio = audio.mean(axis=1)
    if sampling_rate is not None and sr != sampling_rate:
        audio = soxr.resample(audio, sr, sampling_rate, quality="VHQ")
        sr = sampling_rate
    if volume_normalize:
        audio = audio_volume_normalize(audio)
    return audio

class __DatasetConfig:
    sample_rate:int
    max_val_duration:int
    train_duration:int
    speaker_duration:int
    hop_length:int
    device:str
    def update(self, 
        sample_rate:int,
        max_val_duration:int,
        train_duration:int,
        hop_length:int,
        device:str,
        speaker_duration:int = 0
    ):
        self.sample_rate = sample_rate
        self.max_val_duration = max_val_duration
        self.train_duration = train_duration
        self.hop_length = hop_length
        self.device = device
        self.speaker_duration = speaker_duration
        

config = __DatasetConfig()

def __split_batch(all_data:list, batch_size:int):
    all_data = [all_data[i: i + batch_size] for i in range(0, len(all_data), batch_size)]
    return [data for data in all_data if len(data) == batch_size]

def get_sample(file:str, is_train:bool, is_speaker:bool, audio_wav:np.ndarray = None) -> Dict[str, Any]:

    wav = audio_wav if audio_wav is not None else load_audio(file, config.sample_rate, volume_normalize=True)
    wav_length = len(wav)
    wav_length = (wav_length // config.hop_length) * config.hop_length

    duration = config.speaker_duration if is_speaker else config.train_duration
    if not is_train and not is_speaker:
        duration = min(wav_length//config.sample_rate, config.max_val_duration)
    segment_length = int(duration * config.sample_rate)
    if segment_length > wav_length:
        wav = np.pad(wav, (0, int(segment_length - wav_length)))
        wav_length = len(wav)
    start_indice = random.randint(0, wav_length - segment_length) if is_train else 0
    end_indice = start_indice + segment_length
    return {
        "wav" : wav, 
        "start" : start_indice,
        "end" : end_indice,
        "length" : wav_length,
    }

def wav2tensor(batch:list, keys:list):
    return torch.tensor(np.array([
        wav[start:end] for wav, start, end in map(itemgetter(*keys), batch)
    ]), dtype=torch.float32).to(config.device)
def update_batch(batch_data:list,  is_train:bool, epoch:int, return_tensor:bool):
    batch = [
        {
            "index" : it["index"],
            ** get_sample(it["file"], is_train, False)
        }  
        for it in batch_data
    ]
    collate_batch = {
        "is_train" : is_train,
        "index" : [b["index"] for b in batch]
    }
    if "singers" in batch_data[0]:
        singers = [ get_sample(
            it["singers"][epoch % len(it["singers"])]["file"], is_train, True
        ) for it in batch_data ]
        for it, speaker in zip(batch, singers):
            for k, v in speaker.items(): it["spk_" + k] = v
        if return_tensor:
            collate_batch["spk_wav"] = wav2tensor(batch, ["spk_wav", "spk_start", "spk_end"])

    if return_tensor:
        collate_batch["wav"] =  wav2tensor(batch, ["wav", "start", "end"])
        return collate_batch
    collate_batch["data"] = batch
    return collate_batch

def __create_datasets(batch_size:int, data_file:str, type:Literal["music", "singer", "vox"] = "music", min_duration:float = 0):
    if os.path.exists(data_file):
        singer_data_file = data_file[:-5] + f"-{batch_size}ex.json" 
        if type == "singer" and os.path.exists(singer_data_file):
            all_data = read_json(singer_data_file)
            return all_data["train"], all_data["eval"]

        all_data = read_json(data_file)
        datasets = all_data["items"]
        if type == "singer":
            singers = all_data["singers"]
            singers_keys = [key for key in list(singers.keys()) if "、" not in key]
            for it in datasets:
                singer = next((s for s in singers_keys if s != it["singer"]), None)
                it["singers"] = singers[singer][0:5]
                singers[singer] = singers[singer][5:] + singers[singer][:5]
                singers_keys.remove(singer)
                singers_keys.append(singer)

        Num.count([it["duration"] for it in datasets], [0, 2.4, 100])
        train_datasets = __split_batch(datasets[:-20], batch_size)
        eval_datasets = __split_batch(datasets[-20:], 1)
        if type == "singer":
            for it in eval_datasets: it[0]["singers"] = it[0]["singers"][:1]
            save_json(singer_data_file, {"train" : train_datasets, "eval": eval_datasets})
            return __create_datasets(batch_size, data_file, type=type, min_duration=min_duration)
        return train_datasets, eval_datasets
    if type == "music" or type == "singer":
        audio_dir = f"/code/data/custom-datasets/kuaou/split-{min_duration}-{type}/"
        split_audio_by_lrc(
            "/code/data/custom-datasets/kuaou/song", 
            audio_dir, 
            min_duration=int(min_duration*1000), 
            is_recreate=False,
            separated_dir= "/code/data/custom-datasets/kuaou/separated" if type == "singer" else None
         )

        json_files = [read_json(os.path.join(audio_dir, file)) for file in os.listdir(audio_dir) if file.endswith(".json")]
        datasets = [{
            "file": os.path.join(audio_dir, f"{data['id']}-{seg['index']}.wav"),
            "name" : data["name"],
            "text" :  seg['text'],
            "type" : "music",
            "duration" :  (seg["end"] - seg["start"])/1000,
        } for data in json_files  for seg in data["segs"]]

        duration = 0
        singers = {} if type == "singer" else None
        for it in datasets: duration += it["duration"]

        random.shuffle(datasets)
        for i, it in enumerate(datasets): 
            it["index"] = i
        singers = None
        if type == "singer":
            singers = {}
            for it in datasets:
                singer = str(it["name"]).split(" - ")[0]
                it["singer"] = singer
                if singer not in singers:singers[singer] = []
                singers[singer].append({ "file": it["file"], "name" : it["name"]  })



        save_json(data_file, {
            "count" : len(datasets),
            "duration" : duration,
            "items":   datasets
        } | ({"singers" : singers} if singers is not None else {}))
        return __create_datasets(batch_size, data_file, type=type, min_duration=min_duration)
    
    datasets = read_jsonl("/code/data/datasets/voxbox_subset/metadata/aishell-3.jsonl")

    duration = 0
    for i, it in enumerate(datasets): 
        it["file"] = "/code/data/datasets/voxbox_subset/extracted/" + str(it["wav_path"]).replace(".flac", ".wav")
        duration += it["duration"]
        it["index"] = i
        
    save_json(data_file, {
        "count" : len(datasets),
        "duration" : duration,
        "items":   datasets
    })
    return __create_datasets(batch_size, data_file, type=type, min_duration=min_duration)

def create_datasets_with_singers(batch_size:int, data_file:str, min_duration:float = 0):
    return __create_datasets(
        batch_size=batch_size,
        data_file=data_file,
        type="singer",
        min_duration=min_duration
    )

def create_datasets(batch_size:int, data_file:str, type:str = "music", min_duration:float = 0):
    return __create_datasets(
        batch_size=batch_size,
        data_file=data_file,
        type=type,
        min_duration=min_duration
    )
    

    