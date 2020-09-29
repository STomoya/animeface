
import copy

import torch
import torch.nn as nn

class EMA:
    '''Exponential Moving Avergae'''
    def __init__(self, init_model, decay=0.999):
        '''
        args
            init_model : nn.Module
                the model.
                please input the "initialized" model
            decay: float (default:0.999)
                the decay used to update the model.
                usually in [0.9, 0.999]
        '''
        self.decay = decay
        self.G_ema = copy.deepcopy(init_model)
        # freeze and eval mode 
        for param in self.G_ema.parameters():
            param.requires_grad = False
        self.G_ema.cpu().eval()

    def update(self, model_running):
        '''update G_ema
        
        args
            model_running: nn.Module
                the running model.
                must be the same model as EMA
        '''

        # running model to cpu
        original_device = next(model_running.parameters()).device
        model_running.cpu()

        # update params
        ema_param = dict(self.G_ema.named_parameters())
        run_param = dict(G.named_parameters())

        for key in ema_param.keys():
            ema_param[key].data.mul_(self.decay).add_(run_param[key], alpha=(1-self.decay))

        # running model to original device
        model_running.to(original_device)

    def __call__(self, *args, **kwargs):
        '''return the model's output
        
        args
            inputs to the model.
            args can be on gpu.
        '''
        # move inputs to cpu before input
        cpu_args, cpu_kwargs = [], {}
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg = arg.cpu()
            cpu_args.append(arg)
        for key, value in kwargs:
            if isinstance(value, torch.Tensor):
                value = value.cpu()
            cpu_kwargs[key] = value
        
        with torch.no_grad():
            return self.G_ema(*cpu_args, **cpu_kwargs)

if __name__ == "__main__":
    from utils import get_device
    import torch.nn as nn
    device = get_device()
    G = nn.Sequential(
        nn.Conv2d(64, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32,  3, 3, padding=1),
        nn.Tanh()
    )
    G.to(device)

    z = torch.randn(3, 64, 4, 4, device=device)
    print(G(z).size())

    G_ema = EMA(G)
    G_ema.update(G)
    print(G_ema(z).size())