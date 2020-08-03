from __future__ import print_function
import os
import sys
import pprint
import random
import time
import datetime
import argparse
import dateutil.tz
import torch
import torchvision.transforms as transforms
from miscc.config import cfg, cfg_from_file
# config environment
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a layout network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/layout.yml', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--gpu', type=str, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # config cfg file
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('Using config:')
    pprint.pprint(cfg)

    # set manual seed
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    # make output_dir
    if cfg.TRAIN.FLAG:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        output_dir = '../output_bbox_gcn/%s_%s_%s' % \
            (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    else:
        output_dir = '{}'.format(cfg.EVAL.OUTPUT_DIR)

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    image_transform = transforms.Compose([
                                         ])
    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        from datasets import TextDataset
        dataset_train = TextDataset(cfg.DATA_DIR, base_size=cfg.TREE.BASE_SIZE,
                                    transform=image_transform, train_set=True)
        dataset_test = TextDataset(cfg.DATA_DIR, base_size=cfg.TREE.BASE_SIZE,
                                   transform=image_transform, train_set=False)
        assert dataset_train
        assert dataset_test
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, collate_fn=dataset_train.collate_fn,
            num_workers=int(cfg.WORKERS))
        dataloader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=False, collate_fn=dataset_test.collate_fn,
            num_workers=int(cfg.WORKERS))
    else:
        from datasets import TextDataset
        dataset_test = TextDataset(cfg.DATA_DIR, base_size=cfg.TREE.BASE_SIZE,
                                   transform=image_transform, train_set=False)
        assert dataset_test
        dataloader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=cfg.EVAL.BATCH_SIZE,
            drop_last=True, shuffle=False, collate_fn=dataset_test.collate_fn,
            num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    from trainer import LayoutTrainer as trainer
    if cfg.TRAIN.FLAG:
        algo = trainer(output_dir, dataloader_train, imsize, dataloader_test)
    else:
        algo = trainer(output_dir, None, imsize, dataloader_test)
    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.evaluate()
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
