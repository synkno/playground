import torch
import torch.nn as nn
import numpy as np
from .io import Stream
from .object import Obj
import typing
import os
import json

def __format_weights(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def size_of_model(model, format_result = True):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return __format_weights(pp) if format_result else pp


def print_tensor(tensor:torch.Tensor, file:str = None):
    if tensor is None: return
    def floats_to_line(lt):
        sa = []
        if isinstance(lt, list) and isinstance(lt[0], float):
            return ", ".join(["{:.4f}".format(x) for x in lt])
        elif isinstance(lt, list):
            sa += [floats_to_line(x) for x in lt]
        return sa
    result = floats_to_line(tensor.cpu().detach().numpy().astype(np.float32).tolist())
    if file is None:
        print(json.dumps(result, indent=4))
        return
    with open(file,'a+',encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        f.write("\n\n")

class Quantization:
    def __init__(self, version:int = 0, bit:int = 8) -> None:
        if bit < 1 or bit > 8:
            raise Exception(f"Invalid bit {bit}.")
        self.version = version
        self.bit = bit
        self.max_value = 2**bit - 1.0
  
    def quantize(self, line:list)->list[int]:
        line = np.array(line, dtype=np.float32)
        min_value = np.min(line)
        max_value = np.max(line)
        scale = (max_value-min_value)/self.max_value
        line = (line - min_value)/scale
        line = np.round(line).astype(np.uint8).tolist()
        #print(line)

        def add_byte(line:list, s:str):
            while(len(s) > 8):
                line.append(int(s[0:8], 2))
                s = s[8:]
            return s
        
        if self.bit < 8:
            new_line = []
            bins = ""
            for i in line:
                bins += format(i, f'#0{self.bit + 2}b')[2:]
                bins = add_byte(new_line, bins)
            if len(bins) > 0: new_line.append(int(bins + "0" * (8 - len(bins)), 2))
            line = new_line
        return line, float(scale), float(min_value)
    def dequantize(self, line:list[int], scale:float, min_value:float):
        def add_byte(line:list, s:str, bit:int):
            while(len(s) > bit):
                line.append(int(s[0:bit], 2))
                s = s[bit:]
            return s
        if self.bit < 8:
            new_line = []
            bins = ""
            for i in line:
                bins += format(i, f'#010b')[2:]
                bins = add_byte(new_line, bins, self.bit)
            if len(bins) > 0 : new_line.append(int(bins, 2))
            line = new_line#[0:length]
        line = np.array(line, dtype=np.float32)
        #print(np.round(line).astype(np.int32).tolist())
        return (line * scale + min_value).tolist()


def get_fileinfo(model_dir:str, file:str):
    return {"file":file, "size" : os.path.getsize(os.path.join(model_dir, file))}

def quantize_conv1d(st:Stream, ln:nn.Conv1d, q:Quantization):
    ww = ln.weight.transpose(1, 2)
    ww = ww.reshape(ww.shape[0] * ww.shape[1], ww.shape[2])
    quantize_weights(st, ww, q)
    quantize_weights(st, ln.bias, q)

def quantize_weight_bias(st:Stream, ln:typing.Union[nn.Linear, nn.Conv1d, nn.LayerNorm], q:Quantization):
    quantize_weights(st, ln.weight, q)
    quantize_weights(st, ln.bias, q)

def quantize_weights(st:Stream, param:torch.Tensor, q:Quantization = None):
    if param is None:
        st.write_uint32(0)
        st.write_uint32(0)
        return
    param = param.cpu().detach().numpy()
    if len(param.shape) > 2: raise Exception("NotSupport!")
    if len(param.shape) == 1:
        param = param.reshape(1, param.shape[0])
    shape = param.shape
    cols = shape[-1]
    rows = shape[0]

    st.write_uint32(rows)
    st.write_uint32(cols)
    if q is None:
        for i in range(rows):
            for f in param[i]:
                st.write_float32(f)
        return None
    
    scales = []
    weights = []
    for i in range(rows):
        line, scale, min_value  = q.quantize(param[i])
        scales += [scale, min_value]
        weights.append(line)
    for x in scales: st.write_float32(x)
    weights = np.array(weights, dtype=np.uint8).flatten().tolist()
    st.write_uint32(len(weights))
    st.write_bytes(weights)
    return scales


def dequantize_conv1d(st:Stream, ln:nn.Conv1d, q:Quantization):
    shape = ln.weight.shape
    ww = ln.weight.reshape(shape[0] * shape[2], shape[1])
    obj = {"ww" : ww}
    dequantize_weights(st, obj, "ww", q)
    ww = obj["ww"].reshape(shape[0], shape[2], shape[1]).transpose(1, 2)
    ln.weight = nn.Parameter(ww.reshape(shape))
    dequantize_weights(st, ln, "bias", q)

def dequantize_weight_bias(st:Stream, ln:typing.Union[nn.Linear, nn.LayerNorm], q:Quantization):
    dequantize_weights(st, ln, "weight", q)
    dequantize_weights(st, ln, "bias", q)

def dequantize_weights(st:Stream, obj:any, key:str, q:Quantization):
    saved_rows = st.read_uint32() 
    saved_cols = st.read_uint32()
    if saved_rows == 0 and saved_cols == 0:
        return
    
    param:nn.Parameter = Obj.get(obj, key)
    if len(param.shape) > 2: raise Exception("NotSupport!")
    shape = param.shape
    if len(param.shape) == 1:
        shape = (1, shape[0])
    cols = shape[-1]
    rows = shape[0]

    if saved_rows != rows or saved_cols != cols:
        raise Exception(f"InvalidShape saved({saved_rows}x{saved_cols}), current({rows}x{cols})")
    if q is None:
        ft = torch.FloatTensor([st.read_float32() for i in range(cols * rows)]).reshape(param.shape)
        Obj.set(obj, key, nn.Parameter(ft))
        return

    scales = [st.read_float32() for i in range(2*rows)]
    weights = np.array(st.read_bytes(st.read_uint32()), dtype=np.uint8).reshape(rows, cols).tolist()
    new_weights = []
    for i in range(rows):
        new_weights.append(q.dequantize(weights[i], scales[i * 2], scales[i * 2 + 1]))
    ft = torch.FloatTensor(new_weights).reshape(param.shape)
    Obj.set(obj, key, nn.Parameter(ft))