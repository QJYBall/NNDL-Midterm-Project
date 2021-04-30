import numpy as np
from data_loader import VOC_Dataset
from itertools import cycle
from torch.utils.data import DataLoader
import torch
from model import *
import os
import torch.nn as nn
import torch.nn.functional as F
from eval_tool import Dice, eval_score
from utils import label2image
import matplotlib.pyplot as plt


def test(FLAGS):

    BATCH_SIZE_TEST = FLAGS.batch_size_test
    NUM_CLASSES = FLAGS.num_classes
    INPUT_WIDTH = FLAGS.input_width
    INPUT_HEIGHT = FLAGS.input_height
    MODEL = FLAGS.unet_model

    if MODEL == "UNet":
        model = U_Net(3, NUM_CLASSES)
    elif MODEL == "R2UNet":
        model = R2U_Net(3, NUM_CLASSES, 2)
    elif MODEL == "Attention_UNet":
        model = AttU_Net(3, NUM_CLASSES)
    elif MODEL == "Attention_R2UNet":
        model = R2AttU_Net(3, NUM_CLASSES, 2)

    test_image = VOC_Dataset("../data_csv/test_small.csv",INPUT_WIDTH,INPUT_HEIGHT)
    test_loader = cycle(DataLoader(test_image, batch_size = BATCH_SIZE_TEST, shuffle=False))

    model.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.model), map_location='cuda'))
    model.cuda()
    model.eval()

    cross_entropy_loss = nn.CrossEntropyLoss().cuda()
    dice_loss = Dice().cuda()

    true_label = torch.LongTensor()
    pred_label = torch.LongTensor()
    ce_loss = 0.0
    dice_score = 0.0

    with torch.no_grad():

        for iter in range(int(len(test_image) / BATCH_SIZE_TEST)):

            image, label = next(test_loader)
            image, label = image.cuda(), label.cuda()

            output = model(image)
            loss = cross_entropy_loss(output, label)
            ce_loss += loss.cpu().item() * BATCH_SIZE_TEST

            pred = F.log_softmax(output,dim=1).argmax(dim=1).squeeze()
            dice = dice_loss(pred, label)
            dice_score += dice.cpu().item() * BATCH_SIZE_TEST

            true_label = torch.cat((true_label, label.data.cpu()), dim=0)
            pred_label = torch.cat((pred_label, pred.data.cpu()), dim=0)

            if iter == 0:
                
                pred = pred.data.cpu().numpy() 
                label = label.data.cpu().numpy() 
                pred_img, label_img = label2image(NUM_CLASSES)(pred, label)    
                            
                for i in range(BATCH_SIZE_TEST):

                    test_img = image[i]
                    test_seg = pred_img[i]
                    test_label = label_img[i]
                    # 反归一化
                    mean = [.485, .456, .406]
                    std = [.229, .224, .225]
                    x = test_img
                    
                    for j in range(3): 
                        x[j] = np.multiply(x[j],std[j])+mean[j]
                    
                    img = np.multiply(x,255).byte()
                    img = img.numpy().transpose((1, 2, 0))

                    fig, ax = plt.subplots(1, 3,figsize=(30,30))
                    ax[0].imshow(img)
                    ax[1].imshow(test_label)
                    ax[2].imshow(test_seg)
                    plt.savefig('res/pic_{}.png'.format(i))
        
        ce_loss /= len(test_image)
        dice_score /= len(test_image)
        PA, MPA, MIoU, FWIoU = eval_score(true_label.numpy(), pred_label.numpy(), NUM_CLASSES)

    print('ce_loss:{:.4f}, dice:{:.4f}, PA:{:.4f}, MPA:{:.4f}, MIoU:{:.4f}, FWIoU:{:.4f}'.format(
        ce_loss, dice_score, 
        PA, MPA, MIoU, FWIoU
    ))
