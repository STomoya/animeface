
import warnings

import torch
from tqdm import tqdm

def get_device(): return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Status:
    def __init__(self, max_iters):
        self.bar = tqdm(total=max_iters)
        self.batches_done = 0
        self.loss = None

    def update(self, loss):
        self.batches_done += 1

        # loss tracking and visualize
        if isinstance(loss, dict):
            if self.loss is None:
                self._init_loss(loss)
            bar_postfix = []
            for key, value in loss.items():
                self.loss[key].append(value)
                bar_postfix.append(f'{key} : {value:.5f}')
            self.bar.set_postfix_str(' '.join(bar_postfix))
        else:
            warnings.warn('"loss" must be a dict. Skipping loss tracking.')

        self.bar.update(1)

    def plot(self, filename='./loss'):
        if self.loss is None:
            warnings.warn('No loss has been tracked.')
            return

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        legends = []
        for key, values in self.loss.items():
            legends.append(key)
            plt.plot(values)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend(legends, loc='upper right')
        plt.tight_layout()
        plt.savefig(filename)

    def _init_loss(self, loss):
        self.loss = {}
        keys = loss.keys()
        for key in keys:
            self.loss[key] = []