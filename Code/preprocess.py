import pandas as pd
import os

file_name = []
with open("data/VOC2012/ImageSets/Segmentation/trainval.txt","r") as f:
    for line in f.readlines():
        file_name.append(line.strip('\n'))


image = []
label = []
for name in file_name:
    image.append(os.path.join("../data/VOC2012/JPEGImages/"+name+".jpg"))
    label.append(os.path.join("../data/VOC2012/SegmentationClass/"+name+".png"))


df = pd.DataFrame({'image':image,'label':label})

df_train = df.sample(frac=0.7)
df_val_test = df[~df.index.isin(df_train.index)]

df_val = df_val_test.sample(frac=0.3)
df_test = df_val_test[~df_val_test.index.isin(df_val.index)]

df_train.to_csv("data_csv/train.csv")
df_val.to_csv("data_csv/validation.csv")
df_test.to_csv("data_csv/test.csv")

# train 7
# validation 1
# test 2

