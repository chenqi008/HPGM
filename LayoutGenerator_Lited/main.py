import os
import sys
import random
import time
import datetime
import argparse
import pprint
import dateutil.tz
import torch
from miscc.utils import mkdir_p
from miscc.logger import get_logger
from miscc.config import cfg, cfg_from_file
# config environment
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)
# print(sys.path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a layout network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/layout_generator.yml', type=str)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--gpu', dest='gpu', type=str, help='set gpu id')
    control_args = parser.parse_args()
    return control_args


def load_dataset():
    from dataset.datasets import LayoutDataset
    if cfg.TRAIN.FLAG:
        dataset_train = LayoutDataset(cfg.DATA_DIR, cfg.INDICATOR_DIR, cfg.DATASET, train_set=True)
        dataset_test = LayoutDataset(cfg.DATA_DIR, cfg.INDICATOR_DIR, cfg.DATASET, train_set=False)
        assert dataset_train
        assert dataset_test
        loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=True, collate_fn=dataset_train.collate_fn)
        loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=False, collate_fn=dataset_test.collate_fn)
        return loader_train, loader_test
    else:
        dataset_test = LayoutDataset(cfg.DATA_DIR, cfg.INDICATOR_DIR, cfg.DATASET, train_set=False)
        assert dataset_test
        dataloader = torch.utils.data.DataLoader(
            dataset_test, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=False, collate_fn=dataset_test.collate_fn)
        return None, dataloader


if __name__ == "__main__":
    # config cfg file
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.GPU = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU
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
    logger = None
    # make output_dir
    if cfg.TRAIN.FLAG:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        if cfg.TRAIN.GENERATOR:
            output_dir = '../output_bbox_gcn/%s_%s_%s_%s' % \
                ("GENERATOR", cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        else:
            output_dir = '../output_bbox_gcn/%s_%s_%s_%s' % \
                ("EVALUATOR", cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(output_dir)
        logger = get_logger(output_dir, 'log', "experiment.log")
        logger.info("Using config:{}".format(cfg))
    else:
        output_dir = '{}'.format(cfg.EVAL.OUTPUT_DIR)
        logger = get_logger(output_dir, 'log', "eval_experiment.log")
        logger.info("Using config:{}".format(cfg))
    dataloader_train, dataloader_test = load_dataset()
    if cfg.TRAIN.GENERATOR:
        from trainer.trainer_generator import LayGenerator as trainer
    else:
        from trainer.trainer_evaluator import LayoutTrainer as trainer
    if cfg.TRAIN.FLAG:
        our_trainer = trainer(output_dir, dataloader_train, dataloader_test, logger)
    else:
        our_trainer = trainer(output_dir, None, dataloader_test, logger)
    start_t = time.time()
    if cfg.TRAIN.FLAG:
        our_trainer.train()
    else:
        our_trainer.evaluate()
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
