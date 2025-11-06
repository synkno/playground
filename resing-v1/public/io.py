import json
import time
import inspect
import os


def read_json(path:str):
    with open(path,'r',encoding='utf-8') as f:
        datasets = json.load(f)
    return datasets


def save_json(path:str, datasets):
    with open(path,'w+',encoding='utf-8') as f:
        json.dump(datasets, f, ensure_ascii=False, indent=4)

def read_str(path:str)->str:
    with open(path,'r',encoding='utf-8') as f:
        return f.read()
def read_bytes(path:str)->bytes:
    with open(path,'rb') as f:
        return f.read()
    
def save_str(path:str, content:str, mode:str = None):
    if mode is None: mode = "w+"
    with open(path,mode,encoding='utf-8',newline="") as f:
        f.write(content)

def log(tag:str, log_file:str = None):
    t = time.localtime()
    name = ""
    try:
        stack = inspect.stack()
        file, line, func = stack[1][1:4]
        name = os.path.basename(file) + f":{line} {func}"
    except:
        pass
    current_time = time.strftime("%H:%M:%S", t)
    s = f"[{current_time}][{name}] {tag}"
    print(s, flush=True)

    if log_file: save_str(log_file, s + "\n", "a+")