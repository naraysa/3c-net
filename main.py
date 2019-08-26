from __future__ import print_function
import argparse
import os
import torch
from model import Model
from video_dataset import Dataset
from test import test
from train import train
from tensorboard_logger import Logger
import options
from center_loss import CenterLoss
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.optim as optim

if __name__ == '__main__':

    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    
    t_max = 750
    t_max_ctc = 2800
    if args.activity_net:
        t_max = 200
        t_max_ctc = 400
    dataset = Dataset(args)
    
    os.system('mkdir -p ./ckpt/')
    os.system('mkdir -p ./logs/' + args.model_name)
    logger = Logger('./logs/' + args.model_name)
    
    model = Model(dataset.feature_size, dataset.num_class, dataset.labels101to20).to(device)

    if args.eval_only and args.pretrained_ckpt is None:
        print('***************************')
        print('Pretrained Model NOT Loaded')
        print('Evaluating on Random Model')
        print('***************************')

    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt))

    best_acc = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)  
    criterion_cent_f = CenterLoss(num_classes=dataset.num_class, feat_dim=1024, use_gpu=True)
    optimizer_centloss_f = torch.optim.SGD(criterion_cent_f.parameters(), lr=0.1)
    criterion_cent_r = CenterLoss(num_classes=dataset.num_class, feat_dim=1024, use_gpu=True)
    optimizer_centloss_r = torch.optim.SGD(criterion_cent_r.parameters(), lr=0.1)

    criterion_cent_all=[criterion_cent_f, criterion_cent_r]
    optimizer_centloss_all=[optimizer_centloss_f, optimizer_centloss_r]

    for itr in range(args.max_iter):
        dataset.t_max = t_max
        if itr % 2 == 0 and itr > 000:
            dataset.t_max = t_max_ctc
        if not args.eval_only:
            train(itr, dataset, args, model, optimizer, criterion_cent_all, optimizer_centloss_all, logger, device)
          
        if itr % 500 == 0 and (not itr == 0 or args.eval_only):
            acc = test(itr, dataset, args, model, logger, device)
            print(args.summary)
            if acc > best_acc and not args.eval_only:
              torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
              best_acc = acc
        if args.eval_only:
            print('Done Eval!')
            break
