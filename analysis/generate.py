"""
Generates text from a language model
"""
import pickle
import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

def top_p_logits(logits, p=0.9):
    """
    Masks everything but the top probability entries as -infinity.
    """
    if p == 1:
        return logits
    else:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)

        cumprobs = sorted_probs.cumsum(dim=-1)
        # Create mask for all cumulative probabilities less than p
        mask = cumprobs < p
        # First mask must always be pickable
        mask = F.pad(mask[:, :-1], (1, 0, 0, 0), value=1)

        masked_probs = torch.where(mask, sorted_probs, torch.tensor(float('inf')).to(probs))

        batch_mins = masked_probs.min(dim=-1, keepdim=True)[0].expand_as(logits)

        # Mask out all logits (tail) that are too small
        return torch.where(probs < batch_mins, torch.tensor(float('-inf')).to(logits), logits)

def sample_sequence(model, length, batch_size=None, context=None, temperature=1, top_p=0.9, device='cuda', sample=True, eos_token=None):
    assert context is not None

    if not torch.is_tensor(context):
        context = torch.tensor(context, device=device, dtype=torch.long)
    if len(context.size()) < 2:
        context = context.unsqueeze(0).repeat(batch_size, 1)
    
    prev = context
    output = context
    logprobs = 0
    mem = None
    with torch.no_grad():
        for i in range(length):
            logits, mem = model(prev, past=mem)
            logits = logits[:, -1, :] / temperature

            logits = top_p_logits(logits, p=top_p)
                
            probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(probs, num_samples=1)
            else:
                _, prev = torch.topk(probs, k=1, dim=-1)

            logprobs += probs.gather(1, prev).squeeze(-1)
            output = torch.cat((output, prev), dim=1)
            
            # Early break
            if prev.size(0) == 1 and prev.item() == eos_token:
                break
    return output, logprobs / output.size(1)

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='pretrained model path to local checkpoint')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default='../data')
    args = parser.parse_args()
    print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='out/cache')
    tokenizer.max_len = int(1e12)

    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='out/cache')

    if args.model_path:
        state = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state)
    
    model.half().to(device)
    model.eval()
    print('Model loaded.')

    if args.prompt:
        context_tokens = [tokenizer.encoder['<|endoftext|>']] + tokenizer.encode(args.prompt)
    else:
        context_tokens = [tokenizer.encoder['<|endoftext|>']]

    if args.length == -1:
        args.length = model.config.n_positions - len(context_tokens)

    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        out, logprobs = sample_sequence(
            model=model,
            length=args.length,
            context=context_tokens,
            batch_size=args.batch_size,
            temperature=args.temperature,
            device=device,
            eos_token=tokenizer.encoder['<|endoftext|>']
        )
        out = out.tolist()
        logprobs = logprobs.tolist()
        for i in range(args.batch_size):
            generated += 1
            text = tokenizer.decode(out[i])
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)
            print('Mean Log Prob:', logprobs[i])
    print("=" * 80)

if __name__ == '__main__':
    run_model()
