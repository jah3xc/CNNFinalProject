"""
This script will convert a directory of tif
into a identical directory of png
"""
from PIL import Image
import sys
import pathlib
import os
from tqdm import trange

d = sys.argv[1]
files = list(pathlib.Path(d).rglob("*"))
for i in trange(len(files)):
    f = str(files[i])
    if f[f.rfind("."):] == ".tif":
        img = Image.open(f)
        img.save(f[:f.rfind(".")] + ".png", "PNG")
        os.remove(f)