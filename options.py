import argparse

parser = argparse.ArgumentParser(description='3C-Net')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=32, help='number of instances in a batch of data (default: 32)')
parser.add_argument('--model-name', default='3cnet', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature-size', default=2048, help='size of feature (default: 2048)')
parser.add_argument('--num-class', type=int, default=101, help='number of classes (default: 101)')
parser.add_argument('--dataset-name', default='Thumos14', help='dataset to train on (default: Thumos14)')
parser.add_argument('--max-seqlen', type=int, default=750, help='maximum sequence length during training (default: 750)')
parser.add_argument('--max-grad-norm', type=float, default=10, help='value loss coefficient (default: 10)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-iter', type=int, default=50000, help='maximum iteration to train (default: 50000)')
parser.add_argument('--summary', default='no summary', help='Summary of expt')
parser.add_argument('--activity-net', action='store_true', default=False, help='ActivityNet v1.2 dataset')
parser.add_argument('--eval-only', action='store_true', default=False, help='Evaluation only performed')

