import os
import random
import logging

import torch
import numpy as np


MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    't5': 't5-base'
}


def read_input_file(args):
    lines = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.slot_label_file), 'r', encoding='utf-8')]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)