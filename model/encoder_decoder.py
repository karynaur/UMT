import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (BridgeConnection, LayerStack,
                          PositionwiseFeedForward, ResidualConnection, clone)


def attention(Q, K, V, mask, dropout = None):
  """
  
  parameters:
      Q,K,V: size(B, H, seq_len, dim//H)
      mask: Mask if there is any 
  
  returns:
      Attended positions: size(B, H, seq_len, dim//H)

  """

  dk = Q.size(-1)
  QK = Q.matmul(K.transpose(-1,-2))
  out = QK / np.sqrt(dk)

  if mask is not None:
    out = out.masked_fill(mask == 0, -float('inf'))

  out = F.softmax(out, dim = -1)
  out = out.matmul(V)

  if dropout is not None:
      out = dropout(out)

  return out


class MultiHeadAttention(nn.Module):

  def __init__(self, emb, head = 8, dout = None):
    
      """
           Initialize class variables 
      """
       
      super(MultiHeadAttention, self).__init__()
      
      self.head = head
      self.emb = emb

      self.to_Q = nn.Linear(emb, emb)
      self.to_K = nn.Linear(emb, emb)
      self.to_V = nn.Linear(emb, emb)
       
      self.dropout = nn.Dropout(dout) 
      
      assert emb % head == 0, "Oops!, emb not divisible by heads"

      self.dim = emb // head

  def forward(self, Q, K,V, mask):
      """
          parameters:
               Q, K, V: size(Bs, H, B)
      """
      Bs, _, _ = Q.shape

      Q, K, V = self.to_Q(Q), self.to_K(K), self.to_V(V)

      Q = Q.view(Bs, -1, self.head, self.dim).transpose(-3, -2)  # (-4, -3*, -2*, -1)
      K = K.view(Bs, -1, self.head, self.dim).transpose(-3, -2)
      V = V.view(Bs, -1, self.head, self.dim).transpose(-3, -2)

      if mask is not None:
        mask = mask.unsqueeze(1)

      Q = attention(Q, K, V, mask, self.dropout)
      Q = Q.transpose(-3,-2).contiguous().view(Bs, self.head, self.emb)

      return Q


class EncoderLayer(nn.Module):
  
   def __init__(self, emd, dout, head, df):
     """
         parameters:
            emb: Embedding dimention of the transformer
     """ 
     super(EncoderLayer).__init__()
     self.residual = clone(ResidualConnection(emb,dout),2)
     self.attention = MultiHeadAttention(emb, H)
     self.feed_forward = PositionwiseFeedForward(emb, df, dout)

   def forward(self, X, src_mask):
     """
         parameters:
            X: size()
     """  

     sublayer0 = lambda X: self.attention(X, X, X, src_mask)
     sublayer1 = self.feed_forward
        
     X = self.residual[0](X, sublayer0)
     X = self.residual[1](X, sublayer1)

     return X

class Enoder(nn.Module):

   def __init__(self, emb, dout, head df, layers):
     super(Enoder, self).__init__()
     self.encoder_layer = clone(EncoderLayer(emb, dout, head, df) layers)
        
   def forward(self, x, src_mask):
     '''
        in:
            x: (B, S, d_model) src_mask: (B, 1, S)
        out:
            # x: (B, S, d_model) which will be used as Q and K in decoder
     '''
   for layer in self.encoder_layers:
       x = layer(x, src_mask)
    
   return x



class DecoderLayer(nn.Module):
  
   def __init__(self, emd, dout, head, df):
     """
         parameters:
            emb: Embedding dimention of the transformer
     """ 
     super(EncoderLayer).__init__()
     self.residual = clone(ResidualConnection(emb,dout),3)
     self.self_att = MultiHeadAttention(emb, H)
     self.enc_att = MultiHeadAttention(emb, H)
     self.feed_forward = PositionwiseFeedForward(emb, df, dout)

   def forward(self, X, menory, src_mask, tgt_mask):
     """
         parameters:
            X: size()
     """  
     def sublayer0(x): return self.self_att(x, x, x, trg_mask)
     def sublayer1(x): return self.enc_att(x, memory, memory, src_mask)
     sublayer2 = self.feed_forward

     x = self.res_layers[0](x, sublayer0)
     x = self.res_layers[1](x, sublayer1)
     x = self.res_layers[2](x, sublayer2)

     return x

class Decder(nn.Module):

   def __init__(self, emb, dout, head df, layers):
     super(Decder, self).__init__()
     self.decoder_layer = clone(EncoderLayer(emb, dout, head, df) layers)
        
   def forward(self, x, memory, src_mask, tgt_mask): 
     '''
        in:
            x: (B, S, d_model) src_mask: (B, 1, S)
        out:
            # x: (B, S, d_model) which will be used as Q and K in decoder
     '''
   for layer in self.decoder_layers:
       x = layer(x, memory, src_mask, tgt_mask)
    
   return x




   



  






