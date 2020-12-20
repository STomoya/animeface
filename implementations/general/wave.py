
import time
import json
import base64
import subprocess
from threading import Thread

import psutil
from h2o_wave import site, data, ui

def _get_gpu_info():
    keys = ('index', 'name', 'memory.total', 'memory.used', 'utilization.gpu')
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader,nounits'.format(','.join(keys))
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    gpus = [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]
    usage = lambda x: float(x)
    mem = lambda used, total: round(int(used) / int(total) * 100, 1)
    infos = [(usage(gpu['utilization.gpu']), mem(gpu['memory.used'], gpu['memory.total'])) for gpu in gpus]
    return infos

class SystemStat(Thread):
    def __init__(self, name='monitor', gpu=True):
        super().__init__()
        self.gpu_is_visible = gpu
        self.is_tracking = True
        self.page = site[f'/{name}']
        self._gen_cards(gpu)

    def run(self):
        tick = 0
        while self.is_tracking:
            tick += 1

            cpu_usage = psutil.cpu_percent(interval=1)
            self.cpu.data.usage = cpu_usage
            self.cpu.plot_data[-1] = [tick, cpu_usage]

            mem_usage = psutil.virtual_memory().percent
            self.mem.data.usage = mem_usage
            self.mem.plot_data[-1] = [tick, mem_usage]

            if self.gpu_is_visible:
                gpus = _get_gpu_info()
                self.gpu.data.usage = gpus[0][0]
                self.gpu.plot_data[-1] = [tick, gpus[0][0]]

                self.gpu_mem.data.usage = gpus[0][1]
                self.gpu_mem.plot_data[-1] = [tick, gpus[0][1]]

            self.page.save()
            time.sleep(1)

    def stop(self):
        self.is_tracking = False

    def _gen_cards(self, gpu):
        def card(box, title, color):
            return ui.small_series_stat_card(
                box=box,
                title=title,
                value='={{usage}}%',
                data=dict(usage=0.0),
                plot_data=data('tick usage', -15),
                plot_category='tick',
                plot_value='usage',
                plot_zero_value=0,
                plot_color=color
            )
        self.cpu = self.page.add('cpu_stat', card('1 1 1 1', 'CPU', '$red'))
        self.mem = self.page.add('mem_stat', card('2 1 1 1', 'Mem', '$blue'))

        if self.gpu_is_visible:
            self.gpu = self.page.add('gpu_stat', card('3 1 1 1', 'GPU', '$green'))
            self.gpu_mem = self.page.add('gpu_mem_stat', card('4 1 1 1', 'GPU Mem', '$yellow'))

class Training:
    def __init__(self, max_iters, num_loss=2, name='monitor'):
        self.page = site[f'/{name}']
        self._gen_cards(max_iters, num_loss)

    def update(self, iter, loss:dict, filename=None, type='jpg'):
        for key, value in loss.items():
            self.loss.data[-1] = (key, iter, value)
        if filename is not None:
            self.image.type = type
            self.image.image = self._read(filename)
        self.page.save()

    def _gen_cards(self, max_iters, num_loss):
        self.loss = self.page.add('loss', ui.plot_card(
            box='1 2 4 5', # top row for SystemStat
            title='Loss',
            data=data('loss_type iter value', -max_iters*num_loss),
            plot=ui.plot([
                ui.mark(type='line', x_scale='linear', x='=iter', x0=0., y='=value', y0=0., color='=loss_type')
            ])
        ))
        self.image = self.page.add('image', ui.image_card(
            box='5 2 4 5',
            title='Samples',
            type='jpg',
            image=self._read('/usr/src/implementations/general/default.jpg')
        ))

    def _read(self, filename):
        with open(filename, 'rb') as fin:
            image = fin.read()
        return base64.b64encode(image).decode('utf-8')

if __name__ == "__main__":
    import math
    system = SystemStat()
    system.start()
    train = Training(100)
    for i in range(100):
        loss = {
            'G' : math.sin(i/10),
            'D' : math.cos(i/10)
        }
        train.update(i, loss)
        time.sleep(0.2)
    system.stop()

