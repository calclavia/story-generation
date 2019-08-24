"""
Calculates the perplexity given a model.
"""
import pickle
import os, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tokenizer import GPT2Tokenizer
from random import randint
from pytorch_transformers import GPT2LMHeadModel, GPT2Config
from data import PromptDataset, TextDataset
from data.util import wp_preprocess, compose

def compute_logprobs(token_tensor, model):
    input_tokens = token_tensor[:, :-1]
    target_tokens = token_tensor[:, 1:]

    logits, _ = model(input_tokens)
    lprobs = torch.log_softmax(logits, dim=-1)
    # Extract the probability of the target token at each position
    lprobs = lprobs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
    return lprobs

def word_level_ppl(target_tokens, lprobs, tokenizer, raw_token=None):
    assert len(target_tokens) == len(lprobs), (len(target_tokens), len(lprobs))

    # Convert BPE lprobs to word lprobs
    word_lprobs = []
    cur_lp = []
    new_add = ''
    i = 0
    start = False

    for token, lp in zip(target_tokens, lprobs):
        # Follow how it's detokenized.
        chars = tokenizer.decoder[token]
        new_add += bytearray([tokenizer.byte_decoder[c] for c in chars]).decode('utf-8', errors=tokenizer.errors)
        cur_lp.append(lp)

        if not start:
            # Wait for end of prompt
            start = '---\n' in new_add
            if start:
                cur_lp = []
                new_add = ''
            continue

        # Reverse preprocessing
        text = new_add
        text = re.sub('"', ' " ', text)
        text = re.sub('(\'|\.|\,|\:|\?|\!|;)', ' \g<1>', text)
        # Fix contraction
        text = text.replace("n 't", " n't")
        text = text.replace('\n', ' <newline> ')
        text = re.sub(' +', ' ', text)
        text = text.replace('. . .', '...')
        # Edge cases
        text = text.replace("ca n't-", "can't-")
        text = text.replace("St .", "St.")
        text = re.sub(r"//www \.(.*) \.(.*)/", r"//www\.\g<1>\.\g<1>\/", text)

        tokens = text.strip().split(' ')

        # Once a new word is starting to be formed, remove the previous one
        if len(tokens) > i + 1:
            # Token length changed, which means new word has been added.
            # Grab all but the last prob (excluding the unformed next word)
            word_lprobs.append(sum(cur_lp[:-1]))
            cur_lp = cur_lp[-1:]
            i += 1

    # Add final token
    word_lprobs.append(sum(cur_lp))

    token_diff = None
    if raw_token is not None:
        token_diff = abs(len(word_lprobs) - len(raw_token))

    word_lprobs = torch.tensor(word_lprobs)
    ppl = torch.exp(-word_lprobs.mean()).item()
    
    if ppl == float('inf'):
        raise Exception('Infinite PPL', raw_token)

    if ppl > 1000:
        print(ppl)
        print(word_lprobs)
        print(len(word_lprobs), len(raw_token))
        
        raise Exception('Large PPL', tokens, raw_token)
    return ppl, token_diff
    
def run_model():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model-path', type=str, help='pretrained model path to local checkpoint')
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='../data')
    parser.add_argument("--test", action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    if args.batch_size == -1:
        args.batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='out/cache')

    if args.model_path:
        state = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state)

    tokenizer = GPT2Tokenizer(os.path.join(args.data_dir, 'gpt2-vocab.json'), os.path.join(args.data_dir, 'gpt2-merges.txt'))
    # Hack to allow tokenizing longer sequences.
    tokenizer.max_len = int(1e12)

    model.half().to(device)
    model.eval()
    print('Model loaded.')

    d_val = PromptDataset(
        os.path.join(args.data_dir, 'writingPrompts/{}.wp_source'.format('test' if args.test else 'valid')),
        os.path.join(args.data_dir, 'writingPrompts/{}.wp_target'.format('test' if args.test else 'valid')),
        wp_preprocess
    )
    d_val_raw = PromptDataset(
        os.path.join(args.data_dir, 'writingPrompts/{}.wp_source'.format('test' if args.test else 'valid')),
        os.path.join(args.data_dir, 'writingPrompts/{}.wp_target'.format('test' if args.test else 'valid'))
    )
    
    print('Data loaded.')

    print('Running evaluation...')
    with torch.no_grad():
        ppls = []
        word_ppls = []
        token_diffs = []
        num_errs = 0

        batch = []
        for sample_id, (text, check_text) in enumerate(zip(d_val, d_val_raw)):
            bpe_tokens = [tokenizer.encoder['<|endoftext|>']] + tokenizer.encode(text)
            # (This limit applies to GPT2)
            bpe_tokens = bpe_tokens[:1025]
            # Pad
            batch.append((bpe_tokens + [0] * (1025 - len(bpe_tokens)), len(bpe_tokens), check_text.split('---\n')[1].split(' ')))

            if len(batch) == args.batch_size or len(word_ppls) == len(d_val) - 1:
                x, x_lens, raw_tokens = zip(*batch)
                token_tensor = torch.tensor(x, dtype=torch.long, device=device)

                # Compute log probs
                lps = compute_logprobs(token_tensor, model)
                token_tensor = token_tensor.cpu().numpy()

                # Compute individually
                for i in range(lps.shape[0]):
                    try:
                        # Mask out some tokens
                        target_tokens = token_tensor[i, 1:x_lens[i]]
                        log_probs = lps[i, :x_lens[i] - 1]
                        ppl, token_diff = word_level_ppl(target_tokens, log_probs.cpu().float().numpy(), tokenizer, raw_tokens[i])
                        token_diffs.append(token_diff)
                        word_ppls.append(ppl)
                        ppls.append(torch.exp(-log_probs.mean()).item())
                    except Exception as e:
                        print('Skipping anomaly.')
                        print(e)
                        num_errs += 1
                print('World Level PPL {:.2f} BPE PPL {:.2f} Diff {:.2f} Done: {:.2f}% Skip {}'.format(
                    np.mean(word_ppls), np.mean(ppls), np.mean(token_diffs),
                    sample_id / len(d_val) * 100, num_errs
                ))
                batch = []

if __name__ == '__main__':
    run_model()
