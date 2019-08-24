import os, random, json, pickle, re
import numpy as np
import torch.utils.data

class PromptDataset(torch.utils.data.Dataset):
    """
    A dataset for Writing Prompts
    """
    def __init__(self, source, target, preprocess=lambda x: x):
        super().__init__()
        self.preprocess = preprocess

        print('Loading writing prompts...')
        with open(source, errors='ignore') as fs:
            with open(target, errors='ignore') as ft:
                self.prompts = list(zip(fs.readlines(), ft.readlines()))
        print('Done.')

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        prompt, story = self.prompts[i]

        # Remove extra annotation on prompt from WP dataset
        prompt = re.sub('\[ (.*) \]', '', prompt)
        text = 'Prompt: ' + prompt.strip() + '\n---\n' + story.strip()
        text = self.preprocess(text)

        # Safe guard < window size
        if text is None:
            print('Error parsing. Resampling text', i)
            return self[random.randint(0, len(self) - 1)]
        return text