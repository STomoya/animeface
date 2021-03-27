
from typing import Iterable

import torch
import torch.nn as nn

class init:
    modules = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
    bn = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    def __init__(self, my_classes=None, names=None):
        '''
        args:
            my_classes: List[nn.Module]
                list of user defined modules
            names: List[str]
                list of parameter names to initialize
        '''
        if isinstance(my_classes, Iterable):
            if self._are_modules(my_classes):
                self.my_classes = tuple(my_classes)
                self.names = names
            else:
                raise Exception('my_classes should be an iterable with subclasses of nn.Module')
        else: self.my_classes, self.names = [], []

    def _are_modules(self, my_classes):
        return all([issubclass(mcls, nn.Module) for mcls in my_classes])
    
    def N01(self, m):
        '''N(0, 1)'''
        if isinstance(m, self.modules):
            m.weight.data.normal_(0., 1.)
            if not m.bias == None:
                m.bias.data.fill_(0.)
        elif len(self.my_classes) > 0 and len(self.names) > 0:
            if isinstance(m, self.my_classes):
                for name in self.names:
                    if hasattr(m, name):
                        getattr(m, name).data.normal_(0., 1.)
        else:
            self._init_bn(m)

    def N002(self, m):
        '''N(0, 0.02)'''
        if isinstance(m, self.modules):
            m.weight.data.normal_(0, 0.02)
            if not m.bias == None:
                m.bias.data.fill_(0.)
        elif len(self.my_classes) > 0 and len(self.names) > 0:
            if isinstance(m, self.my_classes):
                for name in self.names:
                    if hasattr(m, name):
                        getattr(m, name).data.normal_(0., 0.02)
        else:
            self._init_bn(m)

    def xavier(self, m):
        '''xavier'''
        if isinstance(m, self.modules):
            nn.init.xavier_normal_(m.weight)
            if not m.bias == None:
                m.bias.data.fill_(0.)
        elif len(self.my_classes) > 0 and len(self.names) > 0:
            if isinstance(m, self.my_classes):
                for name in self.names:
                    if hasattr(m, name):
                        nn.init.xavier_normal_(getattr(m, name))
        else:
            self._init_bn(m)

    def kaiming(self, m):
        '''kaiming'''
        if isinstance(m, self.modules):
            nn.init.kaiming_normal_(m.weight)
            if not m.bias == None:
                m.bias.data.fill_(0.)
        elif len(self.my_classes) > 0 and len(self.names) > 0:
            if isinstance(m, self.my_classes):
                for name in self.names:
                    if hasattr(m, name):
                        nn.init.kaiming_normal_(getattr(m, name))
        else:
            self._init_bn(m)

    def _init_bn(self, m):
        '''init batch norm'''
        if isinstance(m, self.bn):
            if not m.weight == None:
                m.weight.data.fill_(1.)
            if not m.bias == None:
                m.bias.data.fill_(0.)

if __name__=='__main__':

    class MyModule1(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(32, 32))
    
    class MyModule2(nn.Module):
        def __init__(self):
            super().__init__()
            self.aaa = nn.Parameter(torch.zeros(32, 32))

    net = nn.Sequential(
        nn.Conv2d(32, 32, 1),
        nn.Conv2d(32, 32, 1, bias=False),
        nn.Linear(32, 32),
        nn.Linear(32, 32, bias=False),
        nn.ConvTranspose2d(32, 32, 1),
        nn.ConvTranspose2d(32, 32, 1, bias=False),
        MyModule1(),
        MyModule2()
    )

    net.apply(init((MyModule1, MyModule2), ('weight', 'aaa')).kaiming)
    print(net[6].weight[0])
    print(net[7].aaa[0])