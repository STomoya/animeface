
import torch
from implementations.StyleGAN3.model import Generator, Discriminator
from thirdparty.ada.augment import AugmentPipe

class ADA(AugmentPipe):
    def __init__(self,
        interval: int, target_kimg, threshold, batch_size,
        **augment_kwargs):
        super().__init__(**augment_kwargs)
        self._interval   = interval
        self._target_img = target_kimg * 1000
        self._threshold  = threshold
        self._batch_size = batch_size
        self._p_delta = batch_size * interval / self._target_img
        self._num_iter = 0

        self.register_buffer('signsum', torch.zeros([]))

    @torch.no_grad()
    def update_p(self, prob: torch.Tensor):
        self.signsum.copy_(self.signsum + torch.sign(prob).sum())
        self._num_iter += 1

        if self._num_iter == self._interval:
            signmean = self.signsum / (self._batch_size * self._interval)
            adjust = torch.sign(signmean - self._threshold) * self._p_delta
            self.p.copy_(self.p + adjust).clamp_(0., 1.)

            self._num_iter = 0
            self.signsum.fill_(0.)
