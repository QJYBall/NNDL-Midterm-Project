from PIL import Image
from scipy.io import loadmat
import pandas as pd
import numpy as np
from utils import label2image


f = open("../data/VOC-aug/SBD/data.txt","r") 
lines = f.readlines()
l2i = label2image() 

count = 0

for line in lines:
    count += 1
    if count % 100 == 0:
        print(count)
    pic = line[:-1]
    data = loadmat("../data/VOC-aug/SBD/cls/"+pic+".mat")
    d = data['GTcls'][0][0][1]
    im = Image.fromarray(l2i(d,d)[0])
    im.save("../data/VOC-aug/SBD/label/"+pic+".png")
    