try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import string
import random
import hashlib

class Xml:
    @staticmethod
    def from_string(str:str):
        return ET.fromstring(str)
    @staticmethod
    def s(ele:ET.Element, path:str, default:str = ""):
        node = Xml.__find(ele, path.split("."))
        node = node[0] if not node is None and len(node) > 0 else None
        if node is None:
            return default
        if isinstance(node, ET.Element):
            if node.text:
                return node.text
            result = ""
            for child in node:
                if child.text:
                    result += child.text
                elif child.tail:
                    result += child.tail
            return result
        return node
    @staticmethod
    def i(ele, path:str, def_value:int = 0):
        s = Xml.s(ele, path, None)
        if s is None:
            return def_value
        try:
            return int(s)
        except ValueError:
            return def_value
    @staticmethod
    def nodes(ele, path:str):
         node = Xml.__find(ele, path.split("."))
         return node if not node is None and len(node) > 0 and type(node[0]) != str else None
    @staticmethod
    def node(ele, path:str):
        node = Xml.__find(ele, path.split("."))
        return node[0] if not node is None and len(node) > 0 and type(node[0]) != str else None
    
    def __find(ele, pathes:list[str]):
        if ele is None: return None
        elements = [ele]
        for path in pathes:
            if not path: return None 
            
            if path.startswith("@"):
                return [ ele.attrib[path[1:]] for ele in elements if path[1:] in ele.attrib]
            new_elements = []
            for ele in elements:
                new_elements += [child for ele in elements for child in ele  if child.tag == path or path == "*"]
            if len(new_elements) <= 0:
                return None
            elements = new_elements
        return elements

class Obj:
    def get(obj:any, path:str, default:any = None):
        if path is None or len(path.strip()) <= 0:
            return obj 
        if obj is None: return default

        sa = path.split(".")
        for path in sa:
            if type(obj) == dict:
                obj = obj[path] if path in obj else None
            elif type(obj) == list:
                index = int(path) if path.isdigit() else 0
                obj = obj[index] if index >= 0 and index < len(obj) else None
            else:
                obj = getattr(obj, path)
            if obj is None: break
        return obj if not obj is None else default

    def set(obj:any, path:str, v:any):
        if obj is None or path is None: return
        sa = path.split(".")
        last_path = sa.pop()
        obj = Obj.get(obj, ".".join(sa))
        if type(obj) == dict:
            obj[last_path] = v
            return
        setattr(obj, last_path, v)

    def to_str(it):
        t = type(it)
        if t == list:
            strs = [Obj.to_str(iit) for iit in it]
            strs.sort()
            return "\n".join(strs)
        if t == dict:
            keys = list(it.keys())
            keys.sort()
            keys = [ {"k" : k , "v" : Obj.to_str(it[k]) } for k in keys]
            return "\n".join([it["k"] + ": " + it["v"] for it in keys if it["v"]])
        return str(it)
    
    def to_key(it):
        key = Obj.to_str(it)
        return  hashlib.md5(key.encode()).hexdigest()

class Str:
    def day():
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d") #%m/%d/%Y, %H:%M:%S
    def time():
        from datetime import datetime
        return datetime.now().strftime("%H-%M-%S.%f") #%m/%d/%Y, %H:%M:%S
    def render(jinja_tempate:str, data:dict):
        import jinja2
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        def raise_exception(message):
            raise TemplateError(message)
        try:
            jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            jinja_env.globals["raise_exception"] = raise_exception
            tmp = jinja_env.from_string(jinja_tempate)
            compiled_template = tmp.render( data = data)
            return compiled_template
        except Exception as e:
            print(e)
            raise e
    
    def random_chars(length: int = 11 , pre:str = '') -> str:
        all_characters = string.ascii_lowercase + string.digits
        random_number = ''.join(random.choice(all_characters) for _ in range(length))
        return pre + '' + random_number
    
    def get_tensor_str(tensor):
        import numpy as np
        import json
        if tensor is None: return
        def floats_to_line(lt):
            sa = []
            if isinstance(lt, list) and isinstance(lt[0], float):
                return ", ".join(["{:.4f}".format(x) for x in lt])
            elif isinstance(lt, list):
                sa += [floats_to_line(x) for x in lt]
            return sa
        if type(tensor) == list:
            result = [floats_to_line(t.detach().cpu().numpy().astype(np.float32).tolist())  for t in tensor]
        else:
            result =  floats_to_line(tensor.detach().cpu().numpy().astype(np.float32).tolist())
        return json.dumps(result, indent=4)


class Num:
    @staticmethod
    def count(nums:list, bins:list):
        import numpy as np
        data = np.array(nums)
        counts, bin_edges = np.histogram(data, bins=bins)
        msg = ""
        for i in range(len(counts)):
            msg += (f"{bin_edges[i]} - {bin_edges[i+1]}: {counts[i]}\n")
        return msg, counts, bin_edges



