import json
import time
import sys
import inspect
import os
import pickle
import struct
from io import BufferedWriter, BufferedReader
from typing import Union


def read_json(path:str):
    with open(path,'r',encoding='utf-8') as f:
        datasets = json.load(f)
    return datasets
def read_jsonl(path:str):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_yaml(file_path):
    import yaml #pip install pyyaml
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

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

def save_bytes(path:str, content:bytes, mode:str = None):
    if mode is None: mode = "wb+"
    with open(path,mode) as f:
        f.write(content)

def save_obj(path, content:any):
    with open(path, 'wb') as file:
        pickle.dump(content, file)

def read_obj(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def image_url(path:str):
    import imghdr
    import base64
    t = imghdr.what(path)
    if t not in ["png", "gif", "jpeg", "jpg"]:
        raise Exception("This " + t + " is not support!")
    encoded_string = base64.b64encode(read_bytes(path)).decode('utf-8') 
    return f'data:image/{t};base64,{encoded_string}'

def find_file_upwards(folder:str, file_name:str):
    path = os.path.abspath(folder if folder else  "./")
    for i in range(10):
        if os.path.exists(os.path.join(path,  file_name)):
            return read_str(os.path.join(path,  file_name))
        path = os.path.split(path)[0]
        if not path: break
    return None

    

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




class Stream:
    def __init__(self, fd:Union[BufferedWriter, BufferedReader]) -> None:
        self.fd = fd

    def write_uint32(self, v:int):
        self.fd.write(bytes([(v) & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff ])) #小端
    def read_uint32(self):
        bs = self.fd.read(4)
        return (bs[3] << 24) | (bs[2] << 16) | (bs[1] << 8) | (bs[0])
    
    def write_uint16(self, v:int):
        if v > 0xffff:  raise Exception(f"({v}) is too long!")
        self.fd.write(bytes([(v) & 0xff, (v >> 8) & 0xff])) #小端
    def read_uint16(self):
        bs = self.fd.read(2)
        return (bs[1] << 8) | (bs[0])
    
    def write_float32(self, v:float):
        bytes = struct.pack('<f', v) #小端
        self.fd.write(bytes)
    def read_float32(self):
        bs = self.fd.read(4)
        return struct.unpack('<f', bs)[0]
    
    def read_bytes(self, size:int):
        return list(self.fd.read(size))
    def write_bytes(self, data:Union[list, bytes]):
        self.fd.write(bytes(data))

    def read_byte(self):
        ret = self.fd.read(1)
        return ret[0] if ret is not None and len(ret) > 0 else -1
    def write_byte(self, b:int):
        self.fd.write(bytes([b]))

    def write_str256(self, s:str):
        data = s.encode('utf-8')
        if len(data) >= 255:
            raise Exception(f"({s}) is too long!")
        self.fd.write(bytes([len(data)]))
        self.fd.write(data)
    def read_str256(self):
        l = self.read_byte()
        if l < 0: return None
        return self.fd.read(l).decode("utf-8")
    
    def write_str(self, s:str):
        data = s.encode('ascii')
        self.fd.write(data)
        self.write_byte(0)
    def read_str(self):
        data = []
        while True:
            b = self.read_byte()
            if b == 0:break
            data.append(b)
        return bytes(data).decode("ascii")
