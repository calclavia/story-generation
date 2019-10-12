import os, time, gc, json, pickle, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from tensorboardX import SummaryWriter
from data.util import *
from util import *

ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
mse_loss_fn = torch.nn.MSELoss(reduction='none')

def compute_loss(device, model, input_tokens, target_tokens, mask):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)

    logits, _ = model(input_tokens)
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = ce_loss_fn(logits.view(-1, num_logits), target_tokens.view(-1)).float().mean()
    loss = ce_loss

    return loss, ce_loss

def compute_ranking_lp(device, model, tokens, mask, random_shift=False):
    """
    Computes the average likelihood score over each class.
    Args:
        tokens: LongTensor of shape [Batch, Classes, Seq Len]
        mask: ByteTensor of shape [Batch, Classes, Seq Len]
    Returns:
        Tensor of [Batch, Classes]
    """
    num_classes = tokens.size(1)
    tokens = tokens.to(device)
    input_tokens = tokens[:, :, :-1]
    target_tokens = tokens[:, :, 1:]

    # Remove classes dimension
    input_tokens = input_tokens.view(-1, input_tokens.size(-1))

    # Randomize the positional encoding
    position_ids = None
    if random_shift:
        position_ids = torch.arange(0, input_tokens.size(-1), dtype=torch.long, device=input_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_tokens)
        
        # position IDs should be expanded from [seq_len] to [batch, seq_len]
        # TODO: hardcoded length
        rand_shift = torch.randint(low=0, high=1024 - input_tokens.size(-1), size=(position_ids.size(0) // num_classes, 1)).to(device)
        # Each class should have the same random shift for fair comparison.
        rand_shift = rand_shift.repeat(1, num_classes).view(-1, 1)
        position_ids = position_ids + rand_shift

    logits, _ = model(input_tokens, position_ids=position_ids)
    logits = logits.view(-1, num_classes, logits.size(-2), logits.size(-1))

    # Pick the target log probs
    lprobs = torch.log_softmax(logits, dim=-1)
    lprobs = lprobs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1).float()
    
    if mask is not None:
        # Cast mask
        mask = mask[:, :, :-1].to(lprobs)
        # Only select masked tokens
        lprobs *= mask
        # Take average log prob across the sequence s.t we have scores for [Batch, 4]
        lprobs = lprobs.float().sum(dim=-1) / mask.sum(dim=-1)
    else:
        lprobs = lprobs.float().mean(dim=-1)
    return lprobs

def train_step(args, device, model, optimizer, input_tokens, target_tokens, mask):
    model.train()

    loss, ce_loss = compute_loss(device, model, input_tokens, target_tokens, mask)
    optimizer.backward(loss)

    return loss.item(), ce_loss.item()

def train_ranking_step(args, device, model, optimizer, tokens, mask):
    model.train()

    lprobs = compute_ranking_lp(device, model, tokens, mask)
    assert len(lprobs.size()) == 2
    # First item is the right answer. We want to maximize that.
    lprob_correct = torch.log_softmax(lprobs, dim=-1)[:, 0]
    loss = -lprob_correct.mean()
    optimizer.backward(loss)

    return loss.item()

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    args.rank = args.rank * ngpus_per_node + gpu
    print('Setting rank', args.rank)
    
    recon_attempt = 1
    connected = False

    if args.rank != 0:
        # Stall to have rank 0 node go first
        time.sleep(3)

    while not connected:
        try:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
            connected = True
            print('Established connection. Rank:', args.rank)
        except Exception as e:
            # Sometimes the head node launches after the worker, which would cause an issue
            print('Failed to init process group. Retrying...', recon_attempt, e)
            recon_attempt += 1
            time.sleep(10)

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    device = torch.device('cuda', args.gpu)
    torch.cuda.set_device(device)

    print('Loading models...')
    cache_dir = os.path.join(args.out_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    if args.rank == 0:
        save_folder = os.path.join(args.out_dir, args.experiment)

        try:
            os.makedirs(save_folder)
        except FileExistsError as e:
            print('Experiment name already exists!', args.experiment)
            raise e

        t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
        v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)

    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    # Hack to allow tokenizing longer sequences.
    tokenizer.max_len = int(1e12)

    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)

    if args.load:
        if args.load == 'none':
            print('Randomly initializing model weights...')
            model.apply(model.init_weights)
        else:
            print('Loading model weights...')
            model.load_state_dict(torch.load(os.path.join(args.load, 'model_latest.pt'), map_location='cpu'))
            gc.collect()

    if args.rank == 0 and args.model_type:
        # Write config to file
        with open(os.path.join(save_folder, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)

    print('params:', num_params(model))
    print('Done.')

    print('Setup data...')
    # Batch and sequence length schedule
    assert len(args.batch_sizes) == len(args.seq_lens)
    batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens)))
    assert len(batch_schedule) == 2, 'Currently not supporting multiple schedule'
    cur_b_schedule = len(batch_schedule) - 1 if args.switch_time == 0 else 0

    print('Batch schedule', batch_schedule)
    train_loader, val_loader = prepare_dataset(
        args.data_dir, args.dataset, tokenizer,
        batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1],
        batch_schedule[-1][0], batch_schedule[-1][1],
        num_workers=args.workers
    )
    print('Done.')

    if args.swag > 0:
        swag_train_loader, swag_val_loader = prepare_dataset(
            args.data_dir,
            'swag',
            tokenizer,
            16, 64,
            16, 64,
            num_workers=1
        )
        print('SWAG Loaded.')
    if args.synth > 0:
        gpt2_train_loader, _ = prepare_dataset(
            args.data_dir,
            'synth',
            tokenizer,
            16, 128,
            16, 128,
            num_workers=1
        )
        print('Loaded GPT2 samples')

    if args.fp16:
        model = model.half()
    model = model.to(device)

    print('Wrapping models and optimizers...')
    # Apply linear scaling rule to increase batch size for short sequence training.
    lr_schedule = switch_schedule(linear_schedule(args), batch_schedule[cur_b_schedule][0] / batch_schedule[-1][0], int(args.iterations * args.switch_time))
    loss_model, optimizer, scheduler = create_optimizers(model, args, lr_schedule)
    print('Done.')

    # TODO: Somehow restoring the optimizer leads to CUDA illegal memory error.
    # if args.load:
    #     print('Loading optimizer weights...')
    #     optimizer.load_state_dict(torch.load(os.path.join(args.load, 'opt_latest.pt'), map_location='cpu'))
    #     gc.collect()

    print('Begin training iterations')
    save_interval = 1000
    max_val_batches = 1000
    num_iters = 0
    e = 0
    optimizer.zero_grad()

    if args.swag > 0:
        swag_iter = iter(swag_train_loader)
    if args.synth > 0:
        gpt2samples_iter = iter(gpt2_train_loader)

    def val_step(val_loader):
        with torch.no_grad():
            print('Validation loop. Batches:', len(val_loader))
            stats = []
            # Validation
            for i, (input_tokens, target_tokens, mask) in enumerate(val_loader):
                loss, ce_loss = compute_loss(device, loss_model, input_tokens, target_tokens, mask)
                stats.append([loss.item(), math.exp(ce_loss.item())])

                if i > max_val_batches:
                    break
            
            stats = np.mean(stats, axis=0)
            v_writer.add_scalar('loss', stats[0], num_iters)
            v_writer.add_scalar('ppl', stats[1], num_iters)

            if args.swag > 0:
                # Process swag
                correct = 0
                total = 0
                for i, (tokens, mask) in enumerate(swag_val_loader):
                    lprobs = compute_ranking_lp(device, model, tokens, mask)
                    chosen = lprobs.argmax(dim=-1)

                    correct += (chosen == 0).sum().item()
                    total += int(chosen.size(0))
                    
                    if i > max_val_batches:
                        break
                v_writer.add_scalar('acc/swag', correct / total, num_iters)
    
    # TODO: Ideally all nodes should run validation.
    if args.rank == 0:
        val_step(val_loader)

    while num_iters < args.iterations:
        # Run epoch
        st = time.time()

        # Training
        print('Training loop. Batches:', len(train_loader))
        for i, (input_tokens, target_tokens, mask) in enumerate(train_loader):
            # Normal grad step
            optimizer.zero_grad()
            loss, ce_loss = train_step(args, device, loss_model, optimizer, input_tokens, target_tokens, mask)
            optimizer.step()

            if args.synth > 0 and i % args.synth == 0:
                # PF grad step
                optimizer.zero_grad()

                try:
                    real_fake_pair = next(gpt2samples_iter)
                except StopIteration:
                    gpt2samples_iter = iter(gpt2_train_loader)
                    real_fake_pair = next(gpt2samples_iter)

                synth_loss = train_ranking_step(args, device, loss_model, optimizer, real_fake_pair, None)
                optimizer.step()

            if args.swag > 0 and i % args.swag == 0:
                optimizer.zero_grad()
                try:
                    swag_loss = train_ranking_step(args, device, loss_model, optimizer, *next(swag_iter))
                except StopIteration:
                    print('Finished one epoch of swag training.')
                    swag_iter = iter(swag_train_loader)
                    swag_loss = train_ranking_step(args, device, loss_model, optimizer, *next(swag_iter))
                optimizer.step()

            if args.rank == 0:
                lr = scheduler.get_lr()[0] if args.warmup != -1 else optimizer.param_groups[0]['lr']
                # Log to Tensorboard
                t_writer.add_scalar('loss', loss, num_iters)

                if args.synth > 0 and i % args.synth == 0:
                    t_writer.add_scalar('loss/synth', synth_loss, num_iters)

                if args.swag > 0 and i % args.swag == 0:
                    t_writer.add_scalar('loss/swag', swag_loss, num_iters)

                t_writer.add_scalar('ppl', math.exp(ce_loss), num_iters)
                t_writer.add_scalar('lr', lr, num_iters)
                t_writer.add_scalar('iter_time', time.time() - st, num_iters)

            st = time.time()
            end = num_iters >= args.iterations

            if args.warmup != -1:
                scheduler.step()

            if args.rank == 0 and num_iters % save_interval == 0 and num_iters > 0:
                print('Saving model...')
                torch.save(model.state_dict(), os.path.join(save_folder, 'model_{:05d}.pt'.format(num_iters // save_interval)))
                torch.save(model.state_dict(), os.path.join(save_folder, 'model_latest.pt'))
                torch.save(optimizer.state_dict(), os.path.join(save_folder, 'opt_latest.pt'))
                torch.save(scheduler.state_dict(), os.path.join(save_folder, 'scheduler_latest.pt'))

            if end:
                break
            num_iters += 1
        
            if num_iters == int(args.iterations * args.switch_time) and args.switch_time > 0:
                print('Switch to long sequence training')
                cur_b_schedule += 1
                train_loader, val_loader = prepare_dataset(
                    args.dataset_dir, args.dataset_name, tokenizer,
                    batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1],
                    batch_schedule[-1][0], batch_schedule[-1][1]
                )

        if args.rank == 0:
            val_step(val_loader)

            print('Saving model...')
            torch.save(model.state_dict(), os.path.join(save_folder, 'model_val_{:05d}.pt'.format(num_iters // save_interval)))
            torch.save(model.state_dict(), os.path.join(save_folder, 'model_latest.pt'))
            torch.save(optimizer.state_dict(), os.path.join(save_folder, 'opt_latest.pt'))
            torch.save(scheduler.state_dict(), os.path.join(save_folder, 'scheduler_latest.pt'))
        e += 1

    print('Training complete.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    # Default parameters are set based on single GPU training
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[4, 2], help='batch size per GPU. Lists the schedule.')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[512, 1024], help='seq length per sample. Lists the schedule.')
    parser.add_argument('--warmup', type=int, default=1000, help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")
    parser.add_argument('--switch-time', type=float, default=0, help="Percentage of iterations to spend on short sequence training.")
    parser.add_argument('--fp16', action='store_true', help="Train using FP16?")
    parser.add_argument('--model-type', type=str, default=None, help="Type of model to use")
    parser.add_argument('--dataset', type=str, default='wp', help="Dataset to use for training")
    parser.add_argument('--swag', default=0,  type=int, help="Use SWAG dataset as auxiliary task?")
    parser.add_argument('--synth', default=0,  type=int, help="Use synthetic examples as auxiliary task?")

    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--load', type=str, help='path to load model from')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')         
    parser.add_argument('--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers')
    args = parser.parse_args()
    print(args)
    print('Starting experiment:', args.experiment)
    # Each node is expected to have same number of GPUs
    ngpus_per_node = torch.cuda.device_count()
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))