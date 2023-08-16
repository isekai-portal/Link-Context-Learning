from bs4 import BeautifulSoup
from pathlib import Path

class ColorConvert():
    def __init__(self,color_table_path):
        with open(color_table_path, 'r', encoding='utf8') as file:
            html_string = file.read()

        soup = BeautifulSoup(html_string, "html.parser")
        self.en_2_rgb = {}
        self.en_2_hex = {}
        self.cn_2_rgb={}
        self.cn_2_hex={}
        self.rgb_2_hex={}
        self.hex_2_rgb={}
        for tr in soup.find_all("tr"):
            tds = tr.find_all("td")
            en_name = tds[1].text
            cn_name = tds[2].text
            hex = tds[3].text
            rgb_str = tds[4].text
            if rgb_str == 'RGB':
                continue
            rgb=tuple(map(int, rgb_str.split(",")))
            self.en_2_rgb[en_name]=rgb
            self.en_2_hex[en_name]=hex
            self.cn_2_rgb[cn_name]=rgb
            self.cn_2_hex[cn_name]=hex
            self.rgb_2_hex[rgb_str]=hex
            self.hex_2_rgb[hex]=rgb
    
    def get_en_2_rgb(self):
        return self.en_2_rgb
    
    def get_cn_2_rgb(self):
        return self.cn_2_rgb
    
    def get_hex_2_rgb(self):
        return self.hex_2_rgb
    
    def get_en_2_hex(self):
        return self.en_2_hex
    
    def get_cn_2_hex(self):
        return self.cn_2_hex
    
    def get_rgb_2_hex(self):
        return self.rgb_2_hex
    
    def get_en_color_list(self):
        return list(self.en_2_rgb.keys())
    
    def get_cn_color_list(self):
        return list(self.cn_2_rgb.keys())

color_table=Path(__file__).parent/"color_table.html"
ColorConvertor = ColorConvert(color_table)

def is_chinese_or_english(string):
    for char in string:
        if '\u4e00' <= char <= '\u9fff':
            return 'cn'
        elif ('\u0041' <= char <= '\u005a') or ('\u0061' <= char <= '\u007a'):
            return 'en'
    return 'no'

def name_2_rgb(color_name:str):
    type = is_chinese_or_english(color_name)
    rgb = None
    if type == 'cn':
        cn_2_rgb=ColorConvertor.get_cn_2_rgb()
        rgb = cn_2_rgb.get(color_name,None)
    elif type == 'en':
        en_2_rgb=ColorConvertor.get_en_2_rgb()
        rgb = en_2_rgb.get(color_name,None)
    
    if type == 'no' or rgb is None:
        raise ValueError(f"input color_name:{color_name} not support!!")
    return rgb

def name_2_hex(color_name:str):
    type = is_chinese_or_english(color_name)
    hex = None
    if type == 'cn':
        cn_2_hex=ColorConvertor.get_cn_2_hex()
        hex = cn_2_hex.get(color_name,None)
    elif type == 'en':
        en_2_hex=ColorConvertor.get_en_2_hex()
        hex = en_2_hex.get(color_name,None)
    
    if type == 'no' or hex is None:
        raise ValueError(f"input color_name:{color_name} not support!!")
    return hex

def cn_color_list():
    return ColorConvertor.get_cn_color_list()

def en_color_list():
    return ColorConvertor.get_en_color_list()

# print(cn_color_list())
# print(en_color_list())