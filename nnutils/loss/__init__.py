
from nnutils.loss.gan import (
    GANLoss,
    NonSaturatingLoss,
    LSGANLoss,
    WGANLoss,
    HingeLoss
)

from nnutils.loss.penalty import (
    calc_grad,
    gradient_penalty,
    dragan_penalty,
    r1_regularizer,
    r2_regularizer
)

from nnutils.loss.vgg import (
    VGGLoss
)
