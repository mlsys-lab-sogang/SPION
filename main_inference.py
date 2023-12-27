import os
import sys
import logging
import argparse
import random
import math
import json
import time
import itertools
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import redirect_stdout
from config import Config
from models.model_LRA import ModelForSC, ModelForSCDual
from models.dataset_LRA import LRADataset
import pickle
import requests
import json


logger = logging.getLogger(__name__)
device_ids = list(range(torch.cuda.device_count()))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_LRA(model, mat_lst, ds_iter, training_config, task):

    val_acc = []
    attn_time = []
    eval_losses = AverageMeter()
    model.eval()
    init_t = time.time()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _, batch in ds_iter['test']:
            if task == 'lra-retrieval':
                input_ids_0 = batch['input_ids_0'].cuda()
                mask_0 = batch['mask_0'].cuda()
                input_ids_1 = batch['input_ids_1'].cuda()
                mask_1 = batch['mask_1'].cuda()
                #print(mask[0])
                label = batch['label'].cuda()
                outputs, attn_lst = model(input_ids_0, input_ids_1, mask_0, mask_1, label, mat_lst, False)
            else:
                input = batch['input_ids_0'].cuda()
                mask = batch['mask_0'].cuda()
                #print(mask[0])
                label = batch['label'].cuda()
                outputs, attn_lst = model(input,mask,label,mat_lst, False)
            loss = outputs["loss"].mean()
            eval_losses.update(loss.mean())
            acc = outputs["accu"].mean()
            val_acc.append(acc)
       
        total_acc = sum(val_acc) / len(val_acc)

    print("total eval time (s): {}".format((time.time()-init_t)))
    end.record()
    torch.cuda.synchronize()
    print("Evaluation Results")
    print("Loss: %2.5f" % eval_losses.avg)
    print("Accuracy: %2.5f" % total_acc)
    print(f"total steps: {len(val_acc)}")
    print(f"total eval time: {(start.elapsed_time(end))}")
    print(f"eval time: {(start.elapsed_time(end))/len(val_acc)}")
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("all memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
    print(torch.cuda.memory_summary(device=0))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, default="eval",
                        help="train eval")
    parser.add_argument("--checkpoint", type = str, default="test",
                        help="load ./checkpoints/model_name.model to evaluation")
    parser.add_argument("--task", type = str, default="lra-image",
                        help = "lra-listops, lra-retrieval, lra-text, lra-pathfinder32-curv_contour_length_14")
    parser.add_argument('--random', type=int, default=42)
    parser.add_argument('--name', type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    if args.task == 'lra-pathfinder':
        args.task = 'lra-pathfinder32-curv_contour_length_14'


    ### get model config ###
    model_config = Config[args.task]["model"]
    model_config["mixed_precision"] = True
    model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))
    model_config["random_seed"] = args.random
    model_config["task"] = args.task

    training_config = Config[args.task]["training"]

    ### log preparation ###
    log_dir = './logs/log-{}/'.format(args.random)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, args.task)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_path = os.path.join(log_dir,'test.{}3.log'.format(args.name))
    redirect_stdout(open(log_path, 'w'))

    ###  set the random seeds for deterministic results. ####
    SEED = args.random
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    device_ids = list(range(torch.cuda.device_count()))
    model_config['batch_size'] = int(training_config['batch_size']/ len(device_ids))
    print(f"GPU list: {device_ids}")

    print(json.dumps([model_config, training_config], indent = 4))

    ### model preparation ###
    if args.task == "lra-retrieval":
        model = ModelForSCDual(model_config, True)
    else:
        model = ModelForSC(model_config, True)

    model = nn.DataParallel(model, device_ids = device_ids)

    mat_lst_path = f'./pickle/layer_attn-{args.random}/mat_lst_{args.task}_{args.name}.pickle'
    with open(mat_lst_path, 'rb') as f:
        mat_lst = pickle.load(f)
    print(mat_lst[0].shape)
    for l in range(model_config["num_layers"]):
        module_name = f"transformer_{l}"
        transformer_module = getattr(model.module.model, module_name)
        mha_module = getattr(transformer_module, "mha")
        mha_module.reconstruct_for_blocksparse(model_config)

    checkpoint_dir = './checkpoints/checkpoints-{}/{}'.format(args.random, args.task)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, '{}.model'.format(args.name))
    training_config["checkpoint_path"] = checkpoint_path

    model = model.cuda()
    print(model)
    print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
    print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

    ### data preparation ###

    ds_iter = {
        "train":DataLoader(LRADataset(f"../data/lra_processed/{args.task}.train.pickle", True), batch_size = training_config["batch_size"], drop_last = True),
        "dev":enumerate(DataLoader(LRADataset(f"./data/lra_processed/{args.task}.dev.pickle", True), batch_size = training_config["batch_size"], drop_last = True)),
        "test":enumerate(DataLoader(LRADataset(f"./data/lra_processed/{args.task}.test.pickle", False), batch_size = training_config["batch_size"], drop_last = True)),
    }

    accumu_steps = model_config["bz_rate"] if "bz_rate" in model_config else 1
    print(f"accumu_steps={accumu_steps}")
    training_config['accumu_steps'] = accumu_steps

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading the best model from: " + checkpoint_path)
        eval_LRA(model, mat_lst, ds_iter, training_config, args.task)
    else:
        print("NO MODEL!!!")

if __name__ == '__main__':
    main()
