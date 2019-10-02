"""
Calculates the SWAG/Story Cloze accuracy given a model.
"""
import pickle
import os, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from random import randint
from datetime import datetime
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from data.util import prepare_dataset
from train import compute_ranking_lp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='pretrained model path to local checkpoint')
    parser.add_argument('--dataset', type=str, default='swag')
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='out/cache')

    if args.model_path:
        if args.model_path == 'random':
            model.apply(model.init_weights)
        else:
            state = torch.load(args.model_path, map_location='cpu')
            model.load_state_dict(state)

    tokenizer = GPT2Tokenizer(os.path.join(args.data_dir, 'gpt2-vocab.json'), os.path.join(args.data_dir, 'gpt2-merges.txt'))

    model.half().to(device)
    model.eval()
    print('Model loaded.')

    loader = prepare_dataset(args.data_dir, args.dataset, tokenizer, args.batch_size, args.seq_len, args.batch_size, args.seq_len,
                            distributed=False, make_train=False, make_val=not args.test, make_test=args.test)[0]
    print('Data loaded.')

    correct = 0
    total = 0

    outputs = []

    with torch.no_grad():
        for tokens, mask in loader:
            lprobs = compute_ranking_lp(device, model, tokens, mask, random_shift=False)
            chosen = lprobs.argmax(dim=-1)

            total += int(chosen.size(0))

            if args.test:
                print('Collecting results...', total)
                outputs += chosen.tolist()
            else:
                correct += (chosen == 0).sum().item()
                print('Accuracy', correct / total)


if __name__ == '__main__':
    main()