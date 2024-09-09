from torch import nn, tensor
from zeta import SSM

model= SSM()

class MambaBlock(nn.Module):
    def __int__(
        self,
        dim: int,
        in_features: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int

    ):
