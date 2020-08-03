from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: layout
__C.DATASET_NAME = 'layout'
# __C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''
__C.GPU_ID = '0'
__C.CUDA = True
__C.WORKERS = 6
__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64


# Test options
__C.TEST = edict()
__C.TEST.B_EXAMPLE = True
__C.TEST.SAMPLE_NUM = 30000


# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.VIS_COUNT = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.CHECK_POINT_INTERVAL = 100
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.FLAG = True
__C.TRAIN.GRAPH_PRE_NET = ''
__C.TRAIN.GCN = ''
__C.TRAIN.BOX_NET = ''

__C.TRAIN.USE_GCN = True
__C.TRAIN.USE_GT_GRAPH = True
__C.TRAIN.USE_SIZE_AS_INPUT = True

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0
__C.TRAIN.COEFF.UNCOND_LOSS = 0.0
__C.TRAIN.COEFF.COLOR_LOSS = 0.0

__C.TRAIN.COEFF.BBOX_LOSS = 1.0

# EVAL options
__C.EVAL = edict()
__C.EVAL.OUTPUT_DIR = ''
__C.EVAL.GRAPH_PRE_NET = ''
__C.EVAL.GCN = ''
__C.EVAL.BOX_NET = ''
__C.EVAL.BATCH_SIZE = 1

# Draw floor plan
__C.FLOOR_PLAN = edict()
__C.FLOOR_PLAN.WIN_DOOR = True

# Extra options
__C.FURNITURE = False
__C.IMAGE_CHANNEL = 1
__C.ROOM_CLASSES = 9
__C.FIXED_NOISE = False
__C.USE_NOISE = True

__C.GRAPH = edict()
__C.GRAPH.LR = 2e-4 # 0.01
__C.GRAPH.WEIGHT_DECAY = 0.0  # 5e-4

__C.GCN = edict()
__C.GCN.LR = 2e-4 # 0.01
__C.GCN.WEIGHT_DECAY = 0.0  # 5e-4

__C.BBOX = edict()
__C.BBOX.LR = 2e-4 # 0.01
__C.BBOX.WEIGHT_DECAY = 0.0  # 5e-4


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)
