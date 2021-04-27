import os
from itertools import cycle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from data_loader import VOC_Dataset
from torch.utils.data import DataLoader
from model import UNet
from utils import weights_init
from eval_tool import Dice, eval_score


def train(FLAGS):

    BATCH_SIZE_TRAIN = FLAGS.batch_size_train
    BATCH_SIZE_VAL = FLAGS.batch_size_val
    NUM_CLASSES = FLAGS.num_classes
    INPUT_WIDTH = FLAGS.input_width
    INPUT_HEIGHT = FLAGS.input_height

    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=False, model=FLAGS.unet_model).cuda()
    model.apply(weights_init)
    
    train_image = VOC_Dataset("../data_csv/train.csv",INPUT_WIDTH,INPUT_HEIGHT)
    train_loader = cycle(DataLoader(train_image, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2, drop_last=True))

    val_image = VOC_Dataset("../data_csv/val.csv",INPUT_WIDTH,INPUT_HEIGHT)
    val_loader = cycle(DataLoader(val_image, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=2, drop_last=True))
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=FLAGS.learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2),
        weight_decay=2e-5
    )

    cross_entropy_loss = nn.CrossEntropyLoss().cuda()
    dice_loss = Dice().cuda()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # load_model is false when training is started from 0th iteration
    with open(FLAGS.log, 'w') as log:
        log.write('Epoch\tIteration\tCross_Entropy_Loss\n') 

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # initialize summary writer
    writer = SummaryWriter()

    best_score = 0

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        
        model.train()

        train_true_label = torch.LongTensor()
        train_pred_label = torch.LongTensor()
        train_dice = 0.0

        for iteration in range(int(len(train_image) / BATCH_SIZE_TRAIN)):

            image, label = next(train_loader)
            image, label = image.cuda(), label.cuda()
                            
            output = model(image)
            output = F.log_softmax(output,dim=1)
            loss = cross_entropy_loss(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # write to log
            with open(FLAGS.log, 'a') as log:
                log.write('{0}\t{1}\t{2}\n'.format(
                    epoch+1,
                    iteration+1,
                    loss.data.storage().tolist()[0],
                ))

            # write to tensorboard
            writer.add_scalar('Cross Entropy Loss', loss.data.storage().tolist()[0],\
                epoch * (int(len(train_image) / BATCH_SIZE_TRAIN) + 1) + iteration) 
                   
            pred = output.argmax(dim=1).squeeze()
            dice = dice_loss(pred, label)
            train_dice += dice.cpu().item() * BATCH_SIZE_TRAIN

            train_true_label = torch.cat((train_true_label, label.data.cpu()), dim=0)
            train_pred_label = torch.cat((train_pred_label, pred.data.cpu()), dim=0)    
            
        train_dice /= len(train_image)
        t_PA, t_MPA, t_MIoU, t_FWIoU = eval_score(train_true_label.numpy(), train_pred_label.numpy(), NUM_CLASSES)
    
        with open(FLAGS.train, 'a') as eval:   
            eval.write('epoch:{}, dice:{:.4f}, PA:{:.4f}, MPA:{:.4f}, MIoU:{:.4f}, FWIoU:{:.4f}\n'.format(
                epoch+1, train_dice, 
                t_PA, t_MPA, t_MIoU, t_FWIoU
            ))

        writer.add_scalar('Dice', train_dice, epoch+1) 
        writer.add_scalar('PA', t_PA, epoch+1) 
        writer.add_scalar('MPA', t_MPA, epoch+1) 
        writer.add_scalar('MIoU', t_MIoU, epoch+1)         
        writer.add_scalar('FWIoU', t_FWIoU, epoch+1) 
                
        scheduler.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.end_epoch: 

            model.eval()
       
            true_label = torch.LongTensor()
            pred_label = torch.LongTensor()
            ce_loss = 0.0
            dice_score = 0.0

            with torch.no_grad():

                for iteration in range(int(len(val_image) / BATCH_SIZE_VAL)):

                    image, label = next(val_loader)
                    image, label = image.cuda(), label.cuda()

                    output = model(image)
                    output = F.log_softmax(output,dim=1)
                    loss = cross_entropy_loss(output, label)
                    ce_loss += loss.cpu().item() * BATCH_SIZE_VAL

                    pred = output.argmax(dim=1).squeeze()
                    dice = dice_loss(pred, label)
                    dice_score += dice.cpu().item() * BATCH_SIZE_VAL

                    true_label = torch.cat((true_label, label.data.cpu()), dim=0)
                    pred_label = torch.cat((pred_label, pred.data.cpu()), dim=0)
                
                ce_loss /= len(val_image)
                dice_score /= len(val_image)
                PA, MPA, MIoU, FWIoU = eval_score(true_label.numpy(), pred_label.numpy(), NUM_CLASSES)
            
            with open(FLAGS.eval, 'a') as eval:   
                eval.write('epoch:{}, ce_loss:{:.4f}, dice:{:.4f}, PA:{:.4f}, MPA:{:.4f}, MIoU:{:.4f}, FWIoU:{:.4f}\n'.format(
                    epoch+1, 
                    ce_loss, dice_score, 
                    PA, MPA, MIoU, FWIoU
                ))

            score = (MPA + MIoU) / 2

            if score > best_score:
                best_score = score
                print("Epoch", epoch+1, "- model saved")
                torch.save(model.state_dict(), os.path.join('checkpoints', FLAGS.model))
