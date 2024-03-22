#! /usr/bin/env python
# -- coding: utf-8 --
import os
import argparse
import datetime
import torch

import train

from fake_article_dataset import PD_Dataset, PS_Dataset
import MyFunc

from eval import eval
from baseline_models import bert_classifier

parser = argparse.ArgumentParser(description='fk_det_model text classificer')

# data
parser.add_argument('-root-path', type=str, default='./', help='the data directory')
parser.add_argument('-dataset-type', type=str, default='fakenews_article', help='choose dataset to run [options: fakenews_article, sentence, fakenews_tweet]')
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
parser.add_argument('-split-ratio', type=str, default='[0.7, 0.8, 1.0]', help='the split ratio of tr, dev, te sets')
parser.add_argument('-benchmark-path', type=str, default='./benchmarked_data/', help='the benchmark data directory')

# learning
parser.add_argument('-lr', type=float, default=0.00001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=1, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=10, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=15, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('-small_data', action='store_true', default=False, help='whether use small dataset for fast debug')

# model
parser.add_argument('-freeze-bert', action='store_true', default=True, help='freeze bert parameters')
parser.add_argument('-bert-type', type=str, default='bert-base-cased', help='the bert embedding choice') # to be replaced by https://huggingface.co/models?sort=downloads&search=BERTweet
parser.add_argument('-model-type', type=int, default=1, help='different structures of metric model, see document for details')
parser.add_argument('-model-name', type=str, default='bert_classifier', help='different structures of metric model, see document for details')
parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.3]')

# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

# evaluation
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
# Add argument for snapshot directory
parser.add_argument('-snapshot-dir', type=str, default='Med-MMHL/snapshot/', help='directory to save snapshots')

args = parser.parse_args()

# update args and print
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

benchmark_dt_path = args.benchmark_path + args.dataset_type
tr, dev, te = MyFunc.read_benchmark_set(benchmark_dt_path)

if 'sentence' in args.dataset_type:
    tr_dataset = PS_Dataset(args, tr)
    dev_dataset = PS_Dataset(args, dev)
    te_dataset = PS_Dataset(args, te)
else:
    tr_dataset = PD_Dataset(args, tr)
    dev_dataset = PD_Dataset(args, dev)
    te_dataset = PD_Dataset(args, te)

tr_dataloader = torch.utils.data.DataLoader(
    tr_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True
)
dev_dataloader = torch.utils.data.DataLoader(
    dev_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True
)
te_dataloader = torch.utils.data.DataLoader(
    te_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True
)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
if args.model_name == 'bert_classifier': # used for any transformer model
    fk_det_model = bert_classifier.BertClassifier(args)
else:
    raise ValueError('the model_name is set wrongly')

if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    fk_det_model.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    fk_det_model = fk_det_model.cuda()

# train or predict
if args.test:
    print('use test mode')
    eval(te_dataloader, fk_det_model, args)
else:
    print('use train mode')
    try:
        print('start training')
        start_epoch = 1

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_accuracy = train.train(tr_dataloader, dev_dataloader, fk_det_model, args)
            val_loss, val_accuracy = eval(dev_dataloader, fk_det_model, args)

            # Save snapshot after every epoch
            snapshot_dir = args.snapshot_dir
            os.makedirs(snapshot_dir, exist_ok=True)  # Ensure snapshot directory exists
            snapshot_path = os.path.join(snapshot_dir, f'model_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': fk_det_model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy
            }, snapshot_path)
            print(f'Model snapshot saved at epoch {epoch}')

    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')
    print('start testing')
    eval(te_dataloader, fk_det_model, args)
