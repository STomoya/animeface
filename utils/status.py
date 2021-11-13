'''utils for training status'''

import logging
import time, datetime
import subprocess
import pprint
import os
import warnings

import torch
from torch.utils.collect_env import get_pretty_env_info
from tqdm import tqdm

class Status:
    '''Status
    A class for keeping loss
    bar: bool (default: True)
        if True, show bar by tqdm
    log_file: str (default: None)
        path to the log file
        if given, log status to a file
    log_interval: int (default: 1)
        interval for writing to log file
    '''
    def __init__(self,
        max_iters: int,
        bar: bool=True,
        log_file: str=None,
        log_interval: int=1,
        logger_name: str='logger'
    ) -> None:
        if bar:
            self.bar = tqdm(total=max_iters)
        self._max_iters = max_iters
        self._batches_done = 0
        self._loss = None
        self._log_file = log_file
        if log_file is not None:
            logging.basicConfig(
                filename=log_file, filemode='w',
                format='%(asctime)s:%(filename)s:%(levelname)s: %(message)s')
            self._logger = logging.getLogger(logger_name)
            self._logger.setLevel(logging.DEBUG)
        self._log_interval = log_interval
        self._step_start = time.time()

    @property
    def max_iters(self):
        return self._max_iters
    @property
    def batches_done(self):
        return self._batches_done
    @batches_done.setter
    def batches_done(self, value):
        self._batches_done = value

    def print(self, *args, **kwargs):
        '''print function'''
        if hasattr(self, 'bar'):
            tqdm.write(*args, **kwargs)
        else:
            print(*args, **kwargs)
    def log(self, message, level='info'):
        if hasattr(self, '_logger'):
            getattr(self._logger, level)(message)
        else:
            warnings.warn('No Logger. Printing to stdout.')
            self.print(message)

    '''Information loggers'''

    def log_args(self, args):
        dict_args = pprint.pformat(vars(args))
        self.log(f'Command line arguments\n{dict_args}')

    def log_torch(self):
        env = get_pretty_env_info()
        self.log(f'PyTorch environment:\n{env}')

    def log_models(self, *models):
        for model in models:
            self.log(f'Architecture: {model.__class__.__name__}\n{model}')

    def log_gpu(self):
        if torch.cuda.is_available():
            nvidia_smi_output = subprocess.run(
                'nvidia-smi', shell=True,
                capture_output=True, universal_newlines=True)
            self.log(f'\n{nvidia_smi_output.stdout}')
        else:
            self.log('No GPU available on your enviornment.')

    def log_training(self, args, *models):
        '''log information in one function'''
        self.log_args(args)
        self.log_torch()
        self.log_models(*models)

    '''a step'''

    def update(self, **kwargs) -> None:
        '''update status'''
        if self._loss == None:
            self._init_loss(kwargs.keys())

        postfix = []
        for k, v in kwargs.items():
            self._loss[k].append(v)
            postfix.append(f'{k} : {v:.5f}')

        # log
        if self._log_file is not None \
            and self.batches_done % self._log_interval == 0:
            # FIXME: this ETA is not exact.
            duration = time.time() - self._step_start
            eta_sec = int((self.max_iters - self.batches_done) * duration)
            eta = datetime.timedelta(seconds=eta_sec)
            self.log(
                f'STEP: {self.batches_done} / {self.max_iters} INFO: {kwargs} ETA: {eta}')
        if self.batches_done == 0:
            # print gpu on first step
            # for checking memory usage
            self.log_gpu()

        self.batches_done += 1
        self._step_start = time.time()

        if hasattr(self, 'bar'):
            self.bar.set_postfix_str(' '.join(postfix))
            self.bar.update(1)

    def is_end(self):
        return self.batches_done >= self.max_iters

    def load_state_dict(self, state_dict: dict) -> None:
        '''fast forward'''
        # load
        self._loss = state_dict['loss']
        self.batches_done = state_dict['batches_done']
        if self.batches_done > 0:
            # fastforward progress bar if present
            if hasattr(self, 'bar'):
                self.bar.update(self.batches_done)

    def state_dict(self) -> dict:
        return dict(
            loss=self._loss,
            batches_done=self.batches_done)

    def _init_loss(self, keys):
        '''initialize loss'''
        self._loss = {}
        for key in keys:
            self._loss[key] = []

    def plot_loss(self, filename='loss'):
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
        for key, values in self._loss.items():
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
        if self._loss == None:
            return 'Loss untracked.'

        string = [f'Batch : {self.batches_done}\t']
        for k, v in self._loss.items():
            string.append(f'{k} : {v[-1]:.5f}')
        return ' '.join(string)
