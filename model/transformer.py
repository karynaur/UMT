from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Generator
from blocks import (BridgeConnection, FeatureEmbedder, Identity,
                          PositionalEncoder, VocabularyEmbedder)
from encoder_decoder import Encoder, Decoder
"""
cfg:
    emb : dimentiona at each layer
    df : no of internal layers in feed_forward
    d_video : no of raw features in input video
    dout : [0,1]
    heads : no of heads
    layers : no of layers
    unfreeze_word_emb : for word embeddings

"""
class Transformer(nn.Module):

  def __init__(self, dataset, cfg):
    super(Transformer,self).__init__()
    
    self.emb = cfg.emb
    self.df = cfg.df
    self.d_feat = cfg.d_video

    self.src_emb = FeatureEmbedder(self.d_feat,self.emb)
    self.tgt_emb = VocabularyEmbedder(dataset.tgt_vocab_size, self.emb)
    self.pos_emb = PositionalEncoder(self.emb, cfg.dout)
    self.enc = Encoder(self.emb, cgf.dout, cfg.heads, self.df, cfg.layers)
    self.dec = Decoder(self.emb, cgf.dout, cfg.heads, self.df, cfg.layers)
    self.generator = Generator(self.emb, dataset.tgt_vocab_size)
    
    # Xavier initialization of parameters
    for o in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    # inititalise word embeddings
    self.tgt_emb.init_word_embeddings(dataset.train_vocab.vectors, cfg.unfreeze_word_emb)
    
  def forward(self, src:dict, tgt, masks:dict):
    
    src = src['rgb'] + src['flow']
    src_mask = masks['V_mask']

    tgt_mask = masks['C_mask']

    src = self.pos_emb(self.src_emb(src))
    tgt = self.pos_emb(self.src_emb(tgt))
    
    out = self.decoder(tgt, self.encoder(src,src_mask), src_mask, tgt_mask)

    out = self.generator(out)

    return out
