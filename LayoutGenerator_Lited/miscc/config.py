from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C
# Dataset name: layout
__C.DATASET_NAME = 'layout'
__C.CUDA = True
__C.DATA_DIR = ''
__C.GT_DATA_DIR = ''
__C.INDICATOR_DIR = ''
__C.GPU = '0'
__C.CONFIG_NAME = 'MultiLayer'
__C.DATASET = ''

# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 500
__C.TRAIN.TAU = 100.
__C.TRAIN.ROOM_HIDDERN_DIM = 64
__C.TRAIN.SCORE_HIDDERN_DIM = 64
__C.TRAIN.EVALUATOR = ''
__C.TRAIN.GENERATOR_MODEL = ''
__C.TRAIN.SAMPLE_NUM = 1
__C.TRAIN.BATCH = True
__C.TRAIN.MARGIN = 1.0
__C.TRAIN.GENERATOR = False
__C.TRAIN.ROOM_GEN_HIDDERN_DIM = 64
__C.TRAIN.BIDIRECTIONAL = True
__C.TRAIN.CONSTRAINT_RULES = [1, 2, 3, 4, 5]

# EVAL options
__C.EVAL = edict()
__C.EVAL.OUTPUT_DIR = ''
__C.EVAL.MODEL_EVALUATOR = ''
__C.EVAL.MODEL_GENERATOR = ''
__C.EVAL.TEST_INDEX = 1
__C.EVAL.EVAL_METRIC = 1

# EVALUATOR options
__C.EVALUATOR = edict()
__C.EVALUATOR.LR = 0.0001
__C.EVALUATOR.WEIGHT_DECAY = 0.0005

# GENERATOR options
__C.GENERATOR = edict()
__C.GENERATOR.LR = 0.0001
__C.GENERATOR.WEIGHT_DECAY = 0.0005
__C.GENERATOR.ALPHA = 20
__C.GENERATOR.BETA = 0
__C.GENERATOR.LOSS = ''
__C.GENERATOR.MAX_EPOCH = 0

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not b.__contains__(k):
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
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    _merge_a_into_b(yaml_cfg, __C)
