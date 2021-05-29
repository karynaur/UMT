import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, d_model, voc_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, voc_size)

    def forward(self, x):
        '''
        Inputs:
            x: (B, Sc, Dc)
        Outputs:
            (B, seq_len, voc_size)
        '''
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


def subsequent_mask(size):
    '''
    in: size
    out: (1, size, size)
    '''
    mask = torch.ones(1, size, size)
    mask = torch.tril(mask, 0)

    return mask.byte()


def mask(src, trg, pad_idx):
    # masking the padding. src shape: (B, S') -> (B, 1, S')
    src_mask = (src != pad_idx).unsqueeze(1)
    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).type_as(src_mask.data)
        return src_mask, trg_mask
    else:
        return src_mask
