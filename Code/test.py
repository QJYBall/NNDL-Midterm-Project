from data_loader import VOC_Dataset
from torch.utils.data import DataLoader
import torch
from model import UNet
import os
import torch.nn as nn
import torch.nn.functional as F
from eval_tool import Dice
from utils import label2image
import matplotlib.pyplot as plt
from itertools import cycle


def test(FLAGS):

    BATCH_SIZE_TEST = FLAGS.batch_size_test
    NUM_CLASSES = FLAGS.num_classes
    INPUT_WIDTH = FLAGS.input_width
    INPUT_HEIGHT = FLAGS.input_height
    MODEL = FLAGS.unet_model

    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=False, model=MODEL)
    # model.cuda()

    test_image = VOC_Dataset("../data_csv/test.csv",INPUT_WIDTH,INPUT_HEIGHT)
    test_loader = cycle(DataLoader(test_image, batch_size = BATCH_SIZE_TEST, shuffle=False))

    model.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.model),map_location='cpu'))
    model.eval()

    cross_entropy_loss = nn.CrossEntropyLoss()
    dice_loss = Dice()
    
    Img = torch.FloatTensor(BATCH_SIZE_TEST, 3, INPUT_WIDTH, INPUT_HEIGHT)
    Label = torch.LongTensor(BATCH_SIZE_TEST, INPUT_WIDTH, INPUT_HEIGHT)

    dice_score = 0
    ce_loss = 0
    
    for iter in range(int(len(test_image) / BATCH_SIZE_TEST)):
        if iter < 10:

            print(iter)

            image, label = next(test_loader)

            Img.copy_(image)
            Label.copy_(label)

            output = model(Img)
            output = F.log_softmax(output,dim=1)

            loss = cross_entropy_loss(output, Label)
            ce_loss += loss * BATCH_SIZE_TEST

            pred = output.argmax(dim=1).squeeze()
            dice = dice_loss(pred, Label)
            dice_score += dice * BATCH_SIZE_TEST
        
        pred = pred.data.cpu().numpy() 
        label = Label.data.cpu().numpy() 
        pred_img, label_img = label2image(NUM_CLASSES)(pred, label)      

        if iter == 0:                        
            for i in range(BATCH_SIZE_TEST):
                image = Img.data.cpu().numpy()
                test_img = image[i]
                test_seg = pred_img[i]
                test_label = label_img[i]
                # 反归一化
                mean = [.485, .456, .406]
                std = [.229, .224, .225]
                x = test_img
                
                for j in range(3): 
                    x[j]=x[j].mul(std[j])+mean[j]
                
                img = x.mul(255).byte()
                img = img.numpy().transpose((1, 2, 0)) # 原图

                fig, ax = plt.subplots(1, 3,figsize=(30,30))
                ax[0].imshow(test_img)
                ax[1].imshow(test_seg)
                ax[2].imshow(test_label)
                plt.savefig('res/pic_{}.png'.format(i))
    
    print("test-Dice: ", dice_score/len(test_image)*4) 
    print("test-CE: ", ce_loss/len(test_image)) 
