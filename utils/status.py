'''utils for training status'''

import warnings
from tqdm import tqdm

class Status:
    '''Status
    A class for keeping loss
    bar: bool (default: False)
        if True, show bar by tqdm
    '''
    def __init__(self,
        max_iters: int,
        bar: bool=True
    ) -> None:
        if bar:
            self.bar = tqdm(total=max_iters)
        self.batches_done = 0
        self.loss = None

    def update(self, **kwargs) -> None:
        '''update status'''
        if self.loss == None:
            self._init_loss(kwargs.keys())

        postfix = []
        for k, v in kwargs.items():
            self.loss[k].append(v)
            postfix.append(f'{k} : {v:.5f}')

        self.batches_done += 1

        if hasattr(self, 'bar'):
            self.bar.set_postfix_str(' '.join(postfix))
            self.bar.update(1)

    def load_state_dict(self, state_dict: dict) -> None:
        '''fast forward'''
        # load
        self.loss = state_dict['loss']
        self.batches_done = state_dict['batches_done']
        if self.batches_done > 0:
            # fastforward progress bar if present
            if hasattr(self, 'bar'):
                self.bar.update(self.batches_done)

    def state_dict(self) -> dict:
        return dict(
            loss=self.loss,
            batches_done=self.batches_done
        )

    def _init_loss(self, keys):
        '''initialize loss'''
        self.loss = {}
        for key in keys:
            self.loss[key] = []

    def plot_loss(self, filename):
        '''plot loss'''
        try:
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
        except ImportError as ie:
            warnings.warn('Could not import matplotlib.')
            return

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
        plt.close()

    def __str__(self):
        '''return latest loss by string'''
        if self.loss == None:
            return 'Loss untracked.'

        string = [f'Batch : {self.batches_done}\t']
        for k, v in self.loss.items():
            string.append(f'{k} : {v[-1]:.5f}')
        return ' '.join(string)
