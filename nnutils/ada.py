
import torch
from thirdparty.ada.augment import AugmentPipe

class ADA(AugmentPipe):
    def __init__(self,
        batch_size: int, interval: int=4, target_kimg: int=500, threshold: float=0.6,
        **augment_kwargs
    ) -> None:
        if augment_kwargs == {}:
            augment_kwargs = dict(
                xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
                brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        super().__init__(**augment_kwargs)
        self._batch_size = batch_size
        self._interval   = interval
        self._target_img = target_kimg * 1000
        self._threshold  = threshold
        self._p_delta = batch_size * interval / self._target_img
        self._num_iter = 0

        self.register_buffer('signsum', torch.zeros([]))
        self.p.copy_(torch.zeros([]))

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
