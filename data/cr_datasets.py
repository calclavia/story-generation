import os, random, json, pickle, re
import numpy as np
import torch.utils.data
import pandas as pd
from itertools import chain

class SWAGDataset(torch.utils.data.Dataset):
    """
    A dataset for SWAG
    """
    def __init__(self, file_path, use_only_gold_examples=True, preprocess=lambda x: x, test_set=False):
        super().__init__()
        self.preprocess = preprocess
        self.use_only_gold_examples = use_only_gold_examples
        self.test_set = test_set

        print('Loading SWAG ...')
        self.swag = pd.read_csv(file_path)

        if self.use_only_gold_examples and file_path.endswith('train.csv'):
            self.swag = self.swag[self.swag['gold-source'].str.startswith('gold')]

        print('Done.')

    def __len__(self):
        return len(self.swag)

    def __getitem__(self, i):
        row = self.swag.iloc[i]

        premise = row['startphrase'] + ' '
        endings = [row['ending{}'.format(i)] for i in range(4)]

        # Remove extra annotation on prompt from WP dataset
        premise_encoded = self.preprocess(premise)

        if premise_encoded is None:
            print('Error parsing. Resampling', i)
            return self[random.randint(0, len(self) - 1)]

        premise_len = [len(premise_encoded)] * 4

        sample_swag = []
        # pop the correct ending
        correct_complete_example = self.preprocess(premise + endings.pop(row['label']))
        sample_swag.append(correct_complete_example)

        # add other endings
        for e in endings:
            complete_example = self.preprocess(premise + e)
            sample_swag.append(complete_example)
        
        if any(x is None for x in sample_swag):
            print('Error parsing. Resampling', i)
            return self[random.randint(0, len(self) - 1)]
            
        return tuple(zip(sample_swag, premise_len)) # a list of 4 lists of tokens

class ROCSDataset(torch.utils.data.Dataset):
    """
    A dataset for ROCS
    """
    def __init__(self, file_path, preprocess=lambda x: x):
        super().__init__()
        self.preprocess = preprocess

        print('Loading ROCS ...')
        self.dataframe = pd.read_csv(file_path)

        print('Done.')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        row = self.dataframe.iloc[i]

        premise = [row['InputSentence{}'.format(i)] for i in range(1,5)]
        premise = ' '.join(premise) + ' '
        endings = [row['RandomFifthSentenceQuiz{}'.format(i)] for i in (1,2)]

        # Remove extra annotation on prompt from WP dataset
        premise_encoded = self.preprocess(premise)
        
        if premise_encoded is None:
            print('Error parsing. Resampling', i)
            return self[random.randint(0, len(self) - 1)]
            
        premise_len = [len(premise_encoded)] * 2

        sample_rocs = []

        # pop the correct ending
        # if labels are not provided, just use default ordering.
        correct_ending = row['AnswerRightEnding'] - 1 if 'AnswerRightEnding' in row else 0

        sample_rocs = (
            self.preprocess(premise + endings[correct_ending]),
            self.preprocess(premise + endings[1 - correct_ending])
        )

        if any(x is None for x in sample_rocs):
            print('Error parsing. Resampling', i)
            return self[random.randint(0, len(self) - 1)]
            
        return tuple(zip(sample_rocs, premise_len)) # a list of 2 lists of tokens
