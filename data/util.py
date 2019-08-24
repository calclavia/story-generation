import random, re, os
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from .prompt_dataset import *
from .cr_datasets import *
from .text_datasets import *
from unidecode import unidecode
import functools

def compose(*functions):
    """ Executes a list of functions in order """
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

def prepare_dataset(data_dir, dataset_name, tokenizer, train_bsz, train_seq_len, val_bsz, val_seq_len,
                    distributed=True, num_workers=1, make_train=True, make_val=True, make_test=False):
    train_collate_fn = collate_fn_lm
    val_collate_fn = collate_fn_lm

    if dataset_name == 'wp':
        train_collate_fn = collate_fn_masked
        val_collate_fn = collate_fn_masked

        train_preproc = WPPreprocessor(tokenizer, train_seq_len)
        val_preproc = WPPreprocessor(tokenizer, val_seq_len)

        d_train = PromptDataset(
            os.path.join(data_dir, 'writingPrompts/train.wp_source'),
            os.path.join(data_dir, 'writingPrompts/train.wp_target'),
            train_preproc)
        d_val = PromptDataset(
            os.path.join(data_dir, 'writingPrompts/valid.wp_source'),
            os.path.join(data_dir, 'writingPrompts/valid.wp_target'),
            val_preproc)
        d_test = PromptDataset(
            os.path.join(data_dir, 'writingPrompts/test.wp_source'),
            os.path.join(data_dir, 'writingPrompts/test.wp_target'),
            val_preproc)
    elif dataset_name == 'swag':
        train_collate_fn = collate_fn_rank
        val_collate_fn = collate_fn_rank

        d_train = SWAGDataset(
            os.path.join(data_dir, 'swag/train.csv'),
            use_only_gold_examples=True,
            preprocess=SWAGPreprocessor(tokenizer, train_seq_len)
        )

        d_val = SWAGDataset(
            os.path.join(data_dir, 'swag/val.csv'),
            use_only_gold_examples=True,
            preprocess=SWAGPreprocessor(tokenizer, val_seq_len)
        )
    elif dataset_name == 'rocs':
        val_collate_fn = collate_fn_rank

        d_train = None
        #  do not have train data
        d_val = ROCSDataset(
            os.path.join(data_dir, 'rocs/val.csv'),
            preprocess=SWAGPreprocessor(tokenizer, val_seq_len)
        )
        d_test = ROCSDataset(
            os.path.join(data_dir, 'rocs/test.csv'),
            preprocess=SWAGPreprocessor(tokenizer, val_seq_len)
        )
    elif dataset_name == 'bookcorpus':
        preproc = GeneralPreprocessor(tokenizer, val_seq_len)
        dataset = TextDataset(os.path.join(data_dir, 'bookcorpus/bookcorpus_clean.txt'), train_seq_len * 8, preproc)
        d_train = torch.utils.data.Subset(dataset, range(0, int(len(dataset) * 0.9)))
        d_val = torch.utils.data.Subset(dataset, range(int(len(dataset) * 0.9), len(dataset)))
    elif dataset_name == 'adv':
        train_collate_fn = collate_fn_text_pair
        preproc = GeneralPreprocessor(tokenizer, val_seq_len)
        real_dset = JSONLTextDataset(os.path.join(data_dir, 'gpt-2-output-dataset/data/webtext.train.jsonl'), preproc)
        fake_dset = JSONLTextDataset(os.path.join(data_dir, 'gpt-2-output-dataset/data/xl-1542M.train.jsonl'), preproc)
        d_train = SyntheticDataset(real_dset, fake_dset)
        d_val = None
    else:
        raise Exception('Invalid dataset')

    if d_train:
        print('Train dataset size', len(d_train))
    if d_val:
        print('Val dataset size', len(d_val))

    loaders = []

    if make_train:
        loaders.append(data.DataLoader(d_train,
                                sampler=DistributedSampler(d_train) if distributed else None, 
                                batch_size=train_bsz, 
                                pin_memory=True, 
                                num_workers=num_workers, 
                                collate_fn=train_collate_fn) if d_train else None)
    if make_val:
        loaders.append(data.DataLoader(d_val,
                            # sampler=DistributedSampler(d_val), 
                            batch_size=val_bsz, 
                            pin_memory=True, 
                            num_workers=num_workers, 
                            collate_fn=val_collate_fn) if d_val else None)

    if make_test:
        loaders.append(data.DataLoader(d_test,
                            # sampler=DistributedSampler(d_val), 
                            batch_size=val_bsz, 
                            pin_memory=True, 
                            num_workers=num_workers, 
                            collate_fn=val_collate_fn) if d_test else None)
    return loaders

class Preprocessor():
    def __init__(self):
        self.fn = None
    def make_fn(self):
        raise NotImplementedError()
    def __call__(self, x):
        try:
            if self.fn is None:
                self.fn = self.make_fn()
            x = self.fn(x)
            return x
        except Exception as e:
            print('Error in preprocessing', repr(e))
            raise e

class WPPreprocessor(Preprocessor):
    def __init__(self, tokenizer, seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    def make_fn(self):
        return compose(
            wp_preprocess,
            lambda x: [self.tokenizer.encoder['<|endoftext|>']] + self.tokenizer.encode(x) + [self.tokenizer.encoder['<|endoftext|>']],
            random_truncate(self.seq_len),
            lambda text: text if len(text) > 1 else None
        )

class GeneralPreprocessor(Preprocessor):
    def __init__(self, tokenizer, seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    def make_fn(self):
        return compose(
            unidecode,
            self.tokenizer.encode,
            random_truncate(self.seq_len),
            lambda text: text if len(text) == self.seq_len else None
        )

class SWAGPreprocessor(Preprocessor):
    def __init__(self, tokenizer, seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    def make_fn(self):
        return compose(
            unidecode,
            self.tokenizer.encode,
            # Filter out sequences too long
            lambda x: None if len(x) > self.seq_len else x
        )

def random_truncate(window):
    """ Randomly truncates text to window size """
    def f(text):
        if len(text) > window:
            # Randomly truncate the text
            start_idx = random.randint(0, len(text) - window - 1)
            text = text[start_idx:start_idx+window] # to account sep and cls tokens
        return text
    return f

def wp_preprocess(text):
    # Standardize some symbols
    text = text.replace('<newline>', '\n')
    text = text.replace('``', '"')
    text = text.replace("''", '"')
    # Detokenize
    text = re.sub(' +', ' ', text)
    text = re.sub(' (\'|\.|\,|\:|\?|\!|;)', '\g<1>', text)
    text = re.sub('" (.*) "', '"\g<1>"', text)
    text = text.replace(" n't", "n't")
    return text

def collate_fn_lm(samples):
    """ Creates a batch out of samples """
    x = torch.LongTensor(samples)
    return x[:, :-1], x[:, 1:].contiguous(), None
    
def collate_fn_text_pair(samples):
    """ Creates a batch out of samples """
    new_samples = []
    for a, b in samples:
        length = min(len(a), len(b))
        new_samples.append([a[:length], b[:length]])

    x = torch.LongTensor(new_samples)
    return x

def collate_fn_masked(samples):
    """ Creates a batch out of samples """
    max_len = max(map(len, samples))
    # Zero pad mask
    x_mask = torch.ByteTensor([[1] * len(x) + [0] * (max_len - len(x)) for x in samples])
    x = torch.LongTensor([x + [0] * (max_len - len(x)) for x in samples])
    return x[:, :-1], x[:, 1:].contiguous(), x_mask[:, 1:]

def collate_fn_rank(samples):
    """
    Creates a batch out of samples.
    Assumes samples is a list of sample of the form:
    Sample: ([(premise + ending, # of tokens for permise), ... 4x]
    The first item must be the correct ending.

    Returns:
        (premise tensor, endings tensor)
    """
    # Maximum length of tokens
    max_len = max(len(x) for sample in samples for x, _ in sample)
    
    # Mask the premise as well as padding
    x_mask = torch.ByteTensor([
        [[0] * prem_len + [1] * (len(x) - prem_len) + [0] * (max_len - len(x)) for x, prem_len in sample]
        for sample in samples 
    ])

    # Pad all token lists to same length
    x = torch.LongTensor([[x + [0] * (max_len - len(x)) for x, _ in sample] for sample in samples])
    return x, x_mask