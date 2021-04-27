import os
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from data_loader import VOC_Dataset
from torch.utils.data import DataLoader
from model import UNet
from utils import setup_seed, weights_init
from eval_tool import Dice

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = '7'


def train(FLAGS):

    setup_seed(78)

    BATCH_SIZE_TRAIN = FLAGS.batch_size_train
    BATCH_SIZE_VAL = FLAGS.batch_size_val
    NUM_CLASSES = FLAGS.num_classes
    INPUT_WIDTH = FLAGS.input_width
    INPUT_HEIGHT = FLAGS.input_height
    MODEL = FLAGS.unet_model

    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=False, model=MODEL)
    model.apply(weights_init)
    model.cuda()

    train_image = VOC_Dataset("../data_csv/train.csv", INPUT_WIDTH, INPUT_HEIGHT)
    train_loader = cycle(
        DataLoader(train_image, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0, drop_last=True))

    val_image = VOC_Dataset("../data_csv/val.csv", INPUT_WIDTH, INPUT_HEIGHT)
    val_loader = cycle(DataLoader(val_image, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0, drop_last=True))

    optimizer = optim.Adam(model.parameters(),
                           lr=FLAGS.learning_rate,
                           betas=(FLAGS.beta_1, FLAGS.beta_2),
                           weight_decay=1e-4)

    cross_entropy_loss = nn.CrossEntropyLoss().cuda()
    dice_loss = Dice()

    Img = torch.FloatTensor(BATCH_SIZE_TRAIN, 3, INPUT_WIDTH, INPUT_HEIGHT).cuda()
    Label = torch.LongTensor(BATCH_SIZE_TRAIN, INPUT_WIDTH, INPUT_HEIGHT).cuda()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # load_model is false when training is started from 0th iteration
    with open(FLAGS.log, 'w') as log:
        log.write('Epoch\tIteration\tCross_Entropy_Loss\n')

    # initialize summary writer
    writer = SummaryWriter()

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):

        model.train()
        for iteration in range(int(len(train_image) / BATCH_SIZE_TRAIN)):

            image, label = next(train_loader)

            optimizer.zero_grad()

            Img.copy_(image)
            Label.copy_(label)

            output = model(Img)
            output = F.log_softmax(output, dim=1)
            loss = cross_entropy_loss(output, Label)
            loss.backward()

            optimizer.step()

            # write to log
            with open(FLAGS.log, 'a') as log:
                log.write('{0}\t{1}\t{2}\n'.format(
                    epoch,
                    iteration,
                    loss.data.storage().tolist()[0],
                ))

            # write to tensorboard
            writer.add_scalar('Cross Entropy Loss',
                              loss.data.storage().tolist()[0],
                              epoch * (int(len(train_image) / BATCH_SIZE_TRAIN) + 1) + iteration)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == FLAGS.end_epoch:
            model.eval()
            with torch.no_grad():
                dice_score = 0
                for iteration in range(int(len(val_image) / BATCH_SIZE_VAL)):

                    image, label = next(val_loader)

                    Img.copy_(image)
                    Label.copy_(label)

                    output = model(Img)
                    output = F.log_softmax(output, dim=1)

                    loss = cross_entropy_loss(output, Label)

                    pred = output.argmax(dim=1).squeeze()
                    dice = dice_loss(pred, Label)
                    dice_score += dice * BATCH_SIZE_VAL

                print("Epoch", epoch, " - dice: ", dice_score / len(val_image))

        if (epoch + 1) % 10 == 0 or (epoch + 1) == FLAGS.end_epoch:
            torch.save(model.state_dict(), os.path.join('checkpoints', FLAGS.model))
