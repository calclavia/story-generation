import os, random, json
import numpy as np
import torch.utils.data
from unidecode import unidecode

class TextDataset(torch.utils.data.Dataset):
    """
    A dataset for language modeling from a text file.
    """
    def __init__(self, file_path, char_window, preprocess=unidecode):
        super().__init__()
        self.char_window = char_window
        self.preprocess = preprocess
        
        self.num_bytes = os.path.getsize(file_path)

        self.file_path = file_path
        self.file = None

    def __len__(self):
        return self.num_bytes // self.char_window

    def __getitem__(self, i):
        if self.file is None:
            self.file = open(self.file_path, errors='ignore')

        self.file.seek(i * self.char_window)
        text = self.file.read(self.char_window)
        text = self.preprocess(text)

        # Safe guard < window size
        if text is None:
            print('Error parsing. Resampling text', i)
            return self[random.randint(0, len(self) - 1)]
        return text

class JSONLTextDataset(torch.utils.data.Dataset):
    """
    A dataset that samples GPT2 samples
    """
    def __init__(self, file_path, preprocess=unidecode):
        super().__init__()
        self.preprocess = preprocess
        self.file_path = file_path
        self.file = None

        # Track read index
        with open(self.file_path) as f:
            self.index = np.cumsum([0] + list(map(lambda x: len(x), f.readlines()))[:-1])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if self.file is None:
            self.file = open(self.file_path, errors='ignore')

        try:
            self.file.seek(self.index[i])
            text = json.loads(self.file.readline())['text']
            text = self.preprocess(text)
        except Exception as e:
            print('Error reading Json line', e, repr(e))
            text = None
            
        # Safe guard < window size
        if text is None:
            print('Error parsing. Resampling text', i)
            return self[random.randint(0, len(self) - 1)]
        return text

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, real_dset, fake_dset):
        super().__init__()
        self.real_dset = real_dset
        self.fake_dset = fake_dset

    def __len__(self):
        return min(len(self.real_dset), len(self.fake_dset))

    def __getitem__(self, i):
        real = self.real_dset[i]
        fake = self.fake_dset[i]
        return real, fake