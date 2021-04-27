import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from utils import train_transform


class VOC_Dataset(Dataset):
    def __init__(self, data_root_csv, input_width, input_height):
        super(VOC_Dataset, self).__init__()

        self.data = pd.read_csv(data_root_csv)

        self.image_list = list(self.data.iloc[:, 0])
        self.label_list = list(self.data.iloc[:, 1])

        self.transform = train_transform
        self.width = input_width
        self.height = input_height

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = Image.open(self.image_list[index]).convert('RGB')
        label = Image.open(self.label_list[index]).convert('RGB')
        img, label = self.transform(img, label, crop_size=(self.width, self.height))

        return img, label
