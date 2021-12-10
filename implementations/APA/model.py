
import torch
import torch.nn as nn

class APA(nn.Module):
    def __init__(self,
        interval: int=4,
        target_kimg: int=500,
        threshold: float=0.6,
        batch_size: int=None,
        # for switching whether to use APA or not
        # with minimum code change
        disable: bool=False
    ) -> None:
        super().__init__()
        self._interval = interval
        self._target_img = target_kimg*1000
        self._threshold = threshold
        self._disable = disable

        self._batch_size = batch_size
        if batch_size is not None:
            self._p_delta = (batch_size*interval) / self._target_img
        else:
            self._p_delta = None

        self.register_buffer('p', torch.zeros([]))
        self.register_buffer('real_signsum', torch.zeros([]))
        self.register_buffer('fake_signsum', torch.zeros([]))

        self._current_iter = 0

    @torch.no_grad()
    def update_p(self,
        *,
        real_prob: torch.Tensor=None,
        fake_prob: torch.Tensor=None
    ):
        '''update p

        Usage:
            apa = APA(...)
            # use lambda_r
            # lambda_r = E[sign(D(x))]
            apa.update_p(real_prob=real_prob)

            # use lambda_f
            # lambda_f = - E[sign(D(G(z)))]
            apa.update_p(fake_prob=fake_prob)

            # use lambda_rf
            # lambda_rf = (lambda_r + lambda_f) / 2

        Arguments:
            real_prob: torch.Tensor (default: None)
                logits(D(x))
            fake_prob: torch.Tensor (default: None)
                logits(D(G(z)))
        '''

        if self._disable:
            return

        # set batch size from first iter
        use_real = isinstance(real_prob, torch.Tensor)
        use_fake = isinstance(fake_prob, torch.Tensor)

        if self._batch_size is None:
            self._set_p_delta(real_prob.size(0) if use_real else fake_prob.size(0))

        if use_real:
            # sum sign(D(x))
            self.real_signsum.copy_(self.real_signsum + torch.sign(real_prob).sum())
        if use_fake:
            # sum sign(D(G(z)))
            self.fake_signsum.copy_(self.fake_signsum + torch.sign(real_prob).sum())
        self._current_iter += 1

        if self._current_iter == self._interval:
            signmean = 0
            if use_real:
                # E[sign(D(x))]
                signmean = self.real_signsum / (self._batch_size * self._interval)
            if use_fake:
                # -E[sign(D(G(z)))]
                # add to signmean for lambda_rf
                signmean = signmean + self.fake_signsum / (self._batch_size * self._interval)
            if all([use_real, use_fake]):
                signmean = signmean / 2

            # -/+ delta
            adjust = torch.sign(signmean - self._threshold) * self._p_delta
            # adjust
            self.p.copy_(self.p + adjust).clamp_(0., 1.)

            self._current_iter = 0

    def forward(self, real, fake):

        if self._disable:
            return real

        if self._batch_size is None:
            self._set_p_delta(real.size(0))

        device = real.device
        alpha     = torch.ones(self._batch_size, 1, 1, 1, device=device)
        zeros     = torch.zeros_like(alpha)
        condition = torch.rand(self._batch_size, 1, 1, 1, device=device) < self.p
        alpha     = torch.where(condition, alpha, zeros)
        if torch.allclose(alpha, zeros):
            return real
        return fake * alpha + real * (1 - alpha)

    def _set_p_delta(self, batch_size: int):
        self._batch_size = batch_size
        self._p_delta = (self._batch_size*self._interval) / self._target_img
