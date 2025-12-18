import os
import gdown

URL = "https://drive.google.com/file/d/1ZPdQqhW6-V1bh6v30sRrWBAmf3ciRvlm/view?usp=drive_link"
OUT = "mask_rcnn_neuron_0012.h5"

if not os.path.exists(OUT):
    print("Downloading model...")
    gdown.download(URL, OUT, quiet=False)
else:
    print("Model already exists.")
