import numpy as np 
import torchvision.transforms as transforms
import random
import torch
import torch.nn as nn


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias,0)


class RandomCrop(object):

    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, label):
        i, j, h, w = self.get_params(img, self.size)
        return img.crop((j,i,j+w,i+h)),label.crop((j,i,j+w,i+h))


class image2label():

    def __init__(self, num_classes=21):
        # classes = ['background','aeroplane','bicycle','bird','boat',
        #    'bottle','bus','car','cat','chair','cow','diningtable',
        #    'dog','horse','motorbike','person','potted plant',
        #    'sheep','sofa','train','tv/monitor']

        colormap = [[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

        self.colormap = colormap[:num_classes]

        cm2lb = np.zeros(256**3)

        for i, cm in enumerate(self.colormap):
            cm2lb[(cm[0]*256+cm[1])*256+cm[2]] = i
        
        self.cm2lb = cm2lb

    def __call__(self, image):
        image = np.array(image, dtype=np.int64)
        idx = (image[:,:,0]*256 + image[:,:,1])*256 + image[:,:,2]
        label = np.array(self.cm2lb[idx], dtype=np.int64)
        return label


def train_transform(image,label,crop_size=(256,256)):

    crop = RandomCrop(crop_size)
    img2lbl = image2label()

    image, label = crop(image,label)

    tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        
    ])

    image = tfs(image)

    label = img2lbl(label)
    label = torch.from_numpy(label).long()
    return image, label


def colormap(n):

    cmap = np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap


class label2image():

    def __init__(self, num_classes=21):

        self.colormap = colormap(256)[:num_classes].astype('uint8')

    def __call__(self, label_pred,label_true):
        pred = self.colormap[label_pred]
        true = self.colormap[label_true]
        return pred, true
