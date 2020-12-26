
import torch
import torch.nn as nn
from torch.autograd import grad, Variable


'''
Base
'''

class Loss:
    '''base class for every loss'''
    def __init__(self, backward=False, return_all=False):
        '''
        args
            backward : bool (default:False)
                if True, .backward() is called before return
            return_all : bool (default:False)
                if True, returns partial calculation results of the total loss if exists (e.g. real_loss, fake_loss in d_loss)
        '''
        self.backward = backward
        self.return_all = return_all


class AdversarialLoss(Loss):
    '''base class for adversarial loss'''
    def d_loss(self, real_prob, fake_prob):
        '''calculates loss for discriminator

        args
            real_prob : torch.Tensor
                D(x)
            fake_prob : torch.Tensor
                D(G(z))
        '''
        raise NotImplementedError()
    def g_loss(self, fake_prob):
        '''calculates loss for generator

        args
            fake_prob : torch.Tensor
                D(G(z))
        '''
        raise NotImplementedError()


'''
Adversarial Loss
'''

class GANLoss(AdversarialLoss):
    '''original GAN loss'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softplus = nn.Softplus()
    def d_loss(self, real_prob, fake_prob):
        '''Ld = E[log(D(x)) + log(1 - D(G(z)))]'''
        real_loss = self.softplus(- real_prob).mean()
        fake_loss = self.softplus(  fake_prob).mean()

        adv_loss = real_loss + fake_loss

        if self.backward:
            adv_loss.backward(retain_graph=True)

        if self.return_all:
            return adv_loss, real_loss, fake_loss
        return adv_loss
    def g_loss(self, fake_prob):
        '''Lg = E[log(D(G(z)))]'''
        fake_loss = self.softplus(- fake_prob).mean()

        if self.backward:
            fake_loss.backward(retain_graph=True)
        
        return fake_loss


class LSGANLoss(AdversarialLoss):
    '''lsgan loss (a,b,c = 0,1,1)'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.MSELoss()
    def d_loss(self, real_prob, fake_prob):
        '''Ld = 1/2*E[(D(x) - 1)^2] + 1/2*E[D(G(z))^2]'''
        real_loss = self.criterion(real_prob, torch.ones(real_prob.size(), device=real_prob.device))
        fake_loss = self.criterion(fake_prob, torch.zeros(fake_prob.size(), device=fake_prob.device))

        adv_loss = (real_loss + fake_loss) / 2

        if self.backward:
            adv_loss.backward(retain_graph=True)

        if self.return_all:
            return adv_loss, real_loss, fake_loss
        return adv_loss
    def g_loss(self, fake_prob):
        '''Lg = 1/2*E[(D(G(z)) - 1)^2]'''
        fake_loss = self.criterion(fake_prob, torch.ones(fake_prob.size(), device=fake_prob.device))
        fake_loss = fake_loss / 2

        if self.backward:
            fake_loss.backward(retain_graph=True)
        
        return fake_loss

class WGANLoss(AdversarialLoss):
    '''wgan loss'''
    def d_loss(self, real_prob, fake_prob):
        '''Ld = E[D(G(z))] - E[D(x)]'''
        real_loss = - real_prob.mean()
        fake_loss =   fake_prob.mean()

        adv_loss = real_loss + fake_loss

        if self.backward:
            adv_loss.backward(retain_graph=True)

        if self.return_all:
            return adv_loss, real_loss, fake_loss
        return adv_loss
    def g_loss(self, fake_prob):
        '''Lg = -E[D(G(z))]'''
        fake_loss = - fake_prob.mean()

        if self.backward:
            fake_loss.backward(retain_graph=True)

        return fake_loss

class HingeLoss(AdversarialLoss):
    '''hinge loss'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()
    def d_loss(self, real_prob, fake_prob):
        '''Ld = - E[min(0, 1 + D(x))] - E[min(0, -1 - D(G(z)))]'''
        real_loss = self.relu(1. - real_prob).mean()
        fake_loss = self.relu(1. + fake_prob).mean()

        adv_loss = real_loss + fake_loss

        if self.backward:
            adv_loss.backward(retain_graph=True)

        if self.return_all:
            return adv_loss, real_loss, fake_loss
        return adv_loss
    def g_loss(self, fake_prob):
        '''Lg = - E[D(G(z))]'''
        fake_loss = - fake_prob.mean()

        if self.backward:
            fake_loss.backward(retain_graph=True)

        return fake_loss


'''
Regularization
'''



