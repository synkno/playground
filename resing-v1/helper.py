
from public.io import read_json, save_json, read_str, log
import os
from scipy.io.wavfile import read
import numpy as np
import random
import shutil
import re
from pydub import AudioSegment
from pydub.silence import detect_silence


out_dir = "/data/custom-datasets/re-sing-44k/"
def pick_songs():
    os.makedirs(out_dir, exist_ok=True)
    songs = read_json("/data/custom-datasets/kuaou/data.json")

    selected = {}
    for song in songs:
        title = str(song["name"]).split(" - ")
        singer = title[0]
        name = title[1]
        if "、" in singer or "." in singer or "（" in name or " " in name or "，" in name or "+" in name:
            continue
        if singer not in selected: selected[singer] = {}
        if name in selected[singer]: 
            continue
        selected[singer][name] = song["id"]
    
    selected = [[{"singer" : k, "name" : name, "id" : id} for name, id in v.items()] for k, v in selected.items()]
    selected = [random.sample(items, k=5) for items in selected if len(items) > 5]
    
    
    #save_json(out_dir + "data.json", selected)


def split_audio_by_lrc():
    
    separated_dir = "/data/custom-datasets/kuaou/separated"
    music_dir = "/data/custom-datasets/kuaou/song"
    min_duration = 8.0 * 1000
    max_duration = 15.0 * 1000
    data = [x  for items in read_json(os.path.join(out_dir, "data.json")) for x in items ]

    result_dir = os.path.join(out_dir, "wavs/")

    shutil.rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir, exist_ok=True)

    skip_silent = separated_dir is not None
    
    for pi, item in enumerate(data):
        lrc = read_str(os.path.join(music_dir, f"{item['id']}.lrc")).splitlines()
        singing_file = os.path.join(separated_dir, f"{item['id']}_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav")

            
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
            if duration < min_duration or duration > max_duration: continue

            out_file = f"{result_dir}{item['id']}-{len(new_segs)}.wav"

            
            
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
        save_json(f"{result_dir}{item['id']}.json", solo_sing)
        log(f"{(pi + 1)}/{len(data)} processed")
    log("compeleted")

