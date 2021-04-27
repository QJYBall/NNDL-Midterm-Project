import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"]='7'

from train import train
from test import test
from utils import setup_seed

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--batch_size_train', type=int, default=16, help='batch size for train')
parser.add_argument('--batch_size_val', type=int, default=16, help='batch size for val')
parser.add_argument('--batch_size_test', type=int, default=15, help='batch size for test')

parser.add_argument('--input_width', type=int, default=300, help="x dimension of the image")
parser.add_argument('--input_height', type=int, default=300, help="y dimension of the image")

parser.add_argument('--num_classes', type=int, default=21, help='the number of the classes')

parser.add_argument('--learning_rate', type=float, default=0.0001, help="starting learning rate")

parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")

parser.add_argument('--model', type=str, default='model.pth', help="model save")

parser.add_argument('--log', type=str, default='log.txt', help="text file to save training logs")
parser.add_argument('--train', type=str, default='train.txt', help="text file to save train logs")
parser.add_argument('--eval', type=str, default='eval.txt', help="text file to save evaluation logs")

parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=500, help="flag to indicate the final epoch of training")

parser.add_argument('--unet_model', type=str, default="UNet", help="model")

FLAGS = parser.parse_args()

if __name__ == '__main__':
    setup_seed(99)
    train(FLAGS)
    # test(FLAGS)