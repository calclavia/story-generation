"""
Calculates the prompt ranking accuracy given a model.
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
from data import PromptDataset, TextDataset
from data.util import wp_preprocess, compose
from .eval import compute_logprobs

def prompt_accuracy():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        help='pretrained model path to local checkpoint')
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='../data')
    parser.add_argument("--test", action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    num_samples = args.num_samples 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='out/cache')

    if args.model_path:
        if args.model_path == 'random':
            model.apply(model.init_weights)
        else:
            state = torch.load(args.model_path, map_location='cpu')
            try:
                model.load_state_dict(state)
            except:
                print('Failed to load weights strictly. Trying unstrict...')
                model.load_state_dict(state, strict=False)

    tokenizer = GPT2Tokenizer(os.path.join(args.data_dir, 'gpt2-vocab.json'), os.path.join(args.data_dir, 'gpt2-merges.txt'))

    model.half().to(device)
    model.eval()
    print('Model loaded.')

    d_val = PromptDataset(
        os.path.join(args.data_dir, 'writingPrompts/{}.wp_source'.format('test' if args.test else 'valid')),
        os.path.join(args.data_dir, 'writingPrompts/{}.wp_target'.format('test' if args.test else 'valid')),
        wp_preprocess
    )
    
    print('Data loaded.')


    d_len = len(d_val)
    get_indices = list(np.random.randint(d_len, size=num_samples))
    hit = 0
    total = 0
    start = datetime.now()

    for i in get_indices:
        sample_ten = []
        sample_prompts = []
        text = d_val[i]
        sample_ten.append(text)

        # get story
        cur_prompt, cur_story = text.split('---\n')
        sample_prompts.append(cur_prompt)
        # sample 9 separate prompts
        while True:
            sample_nine = list(np.random.randint(d_len, size=10))
            if i not in sample_nine:
                break
        for j in sample_nine:
            wrong_prompt = d_val[j].split('---\n')[0]
            
            wrong_text = 'Prompt: ' + wrong_prompt.strip() + '\n---\n' + cur_story.strip()
            sample_ten.append(wrong_text)
            sample_prompts.append('Prompt: ' + wrong_prompt.strip() + '\n---\n')
        
        print('Running evaluation...')
        with torch.no_grad():
            logls = []
            batch = []

            prompt_lens = [len(tokenizer.encode(p)) for p in sample_prompts]
            # tokenise and create batch
            for text in sample_ten:
                bpe_tokens = tokenizer.encode(text)
                # TODO (This limit applies to GPT2)
                bpe_tokens = bpe_tokens[:1025]
                # Pad
                batch.append((bpe_tokens + [0] * (1025 - len(bpe_tokens)), len(bpe_tokens)))

            x, x_lens = zip(*batch)
            token_tensor = torch.tensor(x, dtype=torch.long, device=device)

            # Compute log probs
            lps = compute_logprobs(token_tensor, model)
            token_tensor = token_tensor.cpu().numpy()

            # Compute individually
            for i in range(lps.shape[0]):
                log_probs = lps[i, prompt_lens[i]:x_lens[i] - 1]
                logls.append(-log_probs.float().mean().item())
            
            if logls[0] == min(logls):
                hit += 1
            total += 1
            batch = []
    
        print('{} - Prompt Ranking accuracy: {}'.format(
            datetime.now() - start, hit / total
        ))

if __name__ == '__main__':
    prompt_accuracy()