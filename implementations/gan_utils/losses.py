
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

class GPRegularization(Loss):
    '''gradient penalty regularization'''
    supported_types = ['gp', '0_gp', 'dragan', '0_simple']
    def __init__(self, gp_type='gp', lambda_=10., *args, **kwargs):
        '''
        args
            gp_type : str (default:'gp')
                gradient penalty type.
                in 'gp', 'gp_0', 'dragan', '0_simple'
            lambda_ : float (default:10.)
                lambda for gradient penalty
        '''
        super().__init__(*args, **kwargs)
        assert gp_type in self.supported_types
        self.gp_type = gp_type
        self.center = 0 if '0' in gp_type else 1
        self.lambda_ = lambda_

        self.__assign_regularization()
    
    def __assign_regularization(self):
        '''assign function by 'gp_type' '''
        if self.gp_type in ['gp', '0_gp', 'dragan']:
            self.calc_regularization = self.__calc_gradient_penalty
        elif self.gp_type == '0_simple':
            self.calc_regularization = self.__calc_simple_zero_center_gradient_penalty

    def __calc_gradient_penalty(self, D, real_image, fake_image):
        '''calculates gradient penalty for
            1-GP, 0-GP, dragan

            Lgp = lambda * E[(||grad(D(x^))|| - center)^2]

            where
            1-GP & 0-GP : x^ = alpha * x + (1 - alpha) * x~
            dragan      : x^ = alpha * x + (1 - alpha) * (x + 0.5 * std(x) * U(0, 1))
            when
            x  : real samples
            x~ : generated samples

        args
            D : torch.nn.Module
                discriminator
            real_image : torch.Tensor
                real image samples
            fake_image : torch.Tensor
                generated image samples
        '''
        alpha = torch.randn(real_image.size(0), 1, 1, 1, device=real_image.device)
        if self.gp_type in ['gp', '0_gp']:
            x_hat = (alpha * real_image + (1 - alpha) * fake_image).requires_grad_(True)
        elif self.gp_type == 'dragan':
            x_hat = (alpha * real_image + (1 - alpha) * (real_image + 0.5 * real_image.std() * torch.rand(real_image.size(), device=real_image.device))).requires_grad_(True)

        d_x_hat = D(x_hat)
        ones = torch.ones(d_x_hat.size(), device=d_x_hat.device).requires_grad_(False)

        gradients = grad(
            outputs=d_x_hat,
            inputs=x_hat,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        penalty = (gradients.norm(2, dim=1) - self.center).pow(2).mean()
        penalty = penalty * self.lambda_
        
        if self.backward:
            penalty.backward(retain_graph=True)
        
        return penalty

    def __calc_simple_zero_center_gradient_penalty(self, D, real_image=None, fake_image=None):
        '''calculates gradient penalty for
            simple 0-GP

            Lgp = lambda/2 * R1 + lambda/2 * R2

            where
            R1 = E[(||grad(D(x))||)^2]
            R2 = E[(||grad(D(x~))||)^2]
            when
            x  : real samples
            x~ : generated samples

        args
            D : torch.nn.Module
                discriminator
            real_image : torch.Tensor (default:None)
                if set, R1 regularization is calculated
            fake_image : torch.Tensor (default:None)
                if set, R2 regularization is calculated
        '''
        
        def calc_gp(outputs, inputs):
            '''calc 0-GP'''
            ones = torch.ones(real_prob.size(), device=real_prob.device).requires_grad_(False)
            gradients = grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            penalty = gradients.norm(2, dim=1).pow(2).mean()
            return penalty

        r1_penalty, r2_penalty = 0, 0
        # R1 regularization
        if not real_image == None:
            loc_real = Variable(real_image, requires_grad=True)
            real_prob = D(loc_real).sum()
            r1_penalty = calc_gp(real_prob, loc_real)
            r1_penalty = r1_penalty * self.lambda_ * 0.5
        # R2 regularization
        if not fake_image == None:
            loc_fake = Variable(fake_image, requires_grad=True)
            fake_prob = D(loc_fake).sum()
            r2_penalty = calc_gp(fake_prob, loc_fake)
            r2_penalty = r2_penalty * self.lambda_ * 0.5
        
        penalty = r1_penalty + r2_penalty

        if self.backward:
            penalty.backward(retain_graph=True)

        if self.return_all:
            return penalty, r1_penalty, r2_penalty
        return penalty

if __name__ == "__main__":
    '''TEST'''
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    G = nn.Sequential(
        nn.Conv2d(64, 32, 3, padding=1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(32, 3, 3, padding=1),
        nn.Tanh()
    )
    D = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 1, 3, stride=2, padding=1),
        Flatten()
    )
    z = torch.randn(3, 64, 4, 4)
    x = torch.randn(3,  3, 8, 8)
    # print(D(G(z)).size())

    for gan_loss in [GANLoss, LSGANLoss, WGANLoss, HingeLoss]:
        loss = gan_loss()
        for reg_type in GPRegularization.supported_types:
            print(gan_loss.__name__, reg_type)
            reg = GPRegularization(reg_type)

            # D(x)
            real_prob = D(x)
            # D(G(z))
            fake = G(z)
            fake_prob = D(fake.detach())
            # gan loss
            adv_loss = loss.d_loss(real_prob, fake_prob)
            # regularization
            reg_loss = reg.calc_regularization(D, x, fake)

            d_loss = adv_loss + reg_loss
            d_loss.backward()

            # D(G(z))
            fake = G(z)
            fake_prob = D(fake)
            # gan loss
            g_loss = loss.g_loss(fake_prob)

            g_loss.backward()