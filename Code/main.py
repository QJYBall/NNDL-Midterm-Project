import argparse
from train import train

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--batch_size', type=int, default=16, help='batch size for train')

parser.add_argument('--input_width', type=int, default=256, help="x dimension of the image")
parser.add_argument('--input_height', type=int, default=256, help="y dimension of the image")

parser.add_argument('--num_classes', type=int, default=21, help='the number of the classes')

parser.add_argument('--learning_rate', type=float, default=0.0001, help="starting learning rate")

parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")

parser.add_argument('--model', type=str, default='model.pth', help="model save")

parser.add_argument('--log', type=str, default='log.txt', help="text file to save training logs")

parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=200, help="flag to indicate the final epoch of training")

FLAGS = parser.parse_args()


if __name__ == '__main__':
    train(FLAGS)