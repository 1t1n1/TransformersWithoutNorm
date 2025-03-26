

from itertools import takewhile
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

import torch
print(torch.__version__)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import einops
import wandb
from torchinfo import summary

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    """
    This PE module comes from:
    Pytorch. (2021). LANGUAGE MODELING WITH NN.TRANSFORMER AND TORCHTEXT. https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).to(DEVICE)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(DEVICE)
        pe = torch.zeros(max_len, 1, d_model).to(DEVICE)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = rearrange(x, "b s e -> s b e")
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        x = rearrange(x, "s b e -> b s e")
        return self.dropout(x)

from einops.layers.torch import Rearrange


def attention(
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        mask: torch.BoolTensor=None,
        dropout: nn.Dropout=None,
    ) -> tuple:
    """Computes multihead scaled dot-product attention from the
    projected queries, keys and values.

    Args
    ----
        q: Batch of queries.
            Shape of [batch_size, seq_len_1, n_heads, dim_model].
        k: Batch of keys.
            Shape of [batch_size, seq_len_2, n_heads, dim_model].
        v: Batch of values.
            Shape of [batch_size, seq_len_2, n_heads, dim_model].
        mask: Prevent tokens to attend to some other tokens (for padding or autoregressive attention).
            Attention is prevented where the mask is `True`.
            Shape of [batch_size, n_heads, seq_len_1, seq_len_2],
            or broadcastable to that shape.
        dropout: Dropout layer to use.

    Output
    ------
        y: Multihead scaled dot-attention between the queries, keys and values.
            Shape of [batch_size, seq_len_1, n_heads, dim_model].
        attn: Computed attention between the keys and the queries.
            Shape of [batch_size, n_heads, seq_len_1, seq_len_2].
    """
    qk = torch.einsum('blhd,bmhd->bhlm', q, k)/(q.shape[3]**0.5)
    if mask is not None:
      qk = torch.where(mask, -float('inf'), qk)
    attn = torch.softmax(qk, dim=-1)
    y = torch.einsum('bhlm,bmhd->blhd', attn, v)
    if dropout is not None:
      y = dropout(y)
    return attn, y

class MultiheadAttention(nn.Module):
    """Multihead attention module.
    Can be used as a self-attention and cross-attention layer.
    The queries, keys and values are projected into multiple heads
    before computing the attention between those tensors.

    Parameters
    ----------
        dim: Dimension of the input tokens.
        n_heads: Number of heads. `dim` must be divisible by `n_heads`.
        dropout: Dropout rate.
    """
    def __init__(
            self,
            dim: int,
            n_heads: int,
            dropout: float,
        ):
        super().__init__()

        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)

    def forward(
            self,
            q: torch.FloatTensor,
            k: torch.FloatTensor,
            v: torch.FloatTensor,
            key_padding_mask: torch.BoolTensor = None,
            attn_mask: torch.BoolTensor = None,
        ) -> torch.FloatTensor:
        """Computes the scaled multi-head attention form the input queries,
        keys and values.

        Project those queries, keys and values before feeding them
        to the `attention` function.

        The masks are boolean masks. Tokens are prevented to attends to
        positions where the mask is `True`.

        Args
        ----
            q: Batch of queries.
                Shape of [batch_size, seq_len_1, dim_model].
            k: Batch of keys.
                Shape of [batch_size, seq_len_2, dim_model].
            v: Batch of values.
                Shape of [batch_size, seq_len_2, dim_model].
            key_padding_mask: Prevent attending to padding tokens.
                Shape of [batch_size, seq_len_2].
            attn_mask: Prevent attending to subsequent tokens.
                Shape of [seq_len_1, seq_len_2].

        Output
        ------
            y: Computed multihead attention.
                Shape of [batch_size, seq_len_1, dim_model].
        """
        q = einops.rearrange(self.Wq(q), 'b l (h d)->b l h d', d=self.dim//self.n_heads)
        k = einops.rearrange(self.Wk(k), 'b l (h d)->b l h d', d=self.dim//self.n_heads)
        v = einops.rearrange(self.Wv(v), 'b l (h d)->b l h d', d=self.dim//self.n_heads)
        mask = torch.zeros((q.shape[0], self.n_heads, q.shape[1], k.shape[1]), dtype=bool, device=q.device)
        if attn_mask is not None:
          mask |= attn_mask.reshape(1,1,*attn_mask.shape)
        if key_padding_mask is not None:
            mask |= key_padding_mask.reshape(key_padding_mask.shape[0], 1,1,key_padding_mask.shape[1])
        _,y = attention(q,k,v,mask,self.dropout)
        y = einops.rearrange(y, 'b l h d->b l (h d)')
        return y

class TransformerDecoderLayer(nn.Module):
    """Single decoder layer.

    Parameters
    ----------
        d_model: The dimension of decoders inputs/outputs.
        dim_feedforward: Hidden dimension of the feedforward networks.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            nhead: int,
            dropout: float,
            normcls = nn.LayerNorm
        ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.nhead = nhead

        self.selfattn = MultiheadAttention(d_model, nhead, dropout)
        self.normselfattn = normcls(d_model)
        self.crossattn = MultiheadAttention(d_model, nhead, dropout)
        self.normcrossattn = normcls(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.normff = normcls(d_model)

    def forward(
            self,
            src: torch.FloatTensor,
            tgt: torch.FloatTensor,
            tgt_mask_attn: torch.BoolTensor,
            src_key_padding_mask: torch.BoolTensor,
            tgt_key_padding_mask: torch.BoolTensor,
        ) -> torch.FloatTensor:
        """Decode the next target tokens based on the previous tokens.

        Args
        ----
            src: Batch of source sentences.
                Shape of [batch_size, src_seq_len, dim_model].
            tgt: Batch of target sentences.
                Shape of [batch_size, tgt_seq_len, dim_model].
            tgt_mask_attn: Mask to prevent attention to subsequent tokens.
                Shape of [tgt_seq_len, tgt_seq_len].
            src_key_padding_mask: Mask to prevent attention to padding in src sequence.
                Shape of [batch_size, src_seq_len].
            tgt_key_padding_mask: Mask to prevent attention to padding in tgt sequence.
                Shape of [batch_size, tgt_seq_len].

        Output
        ------
            y:  Batch of sequence of embeddings representing the predicted target tokens
                Shape of [batch_size, tgt_seq_len, dim_model].
        """
        tgt = tgt + self.selfattn(tgt, tgt, tgt, tgt_key_padding_mask, tgt_mask_attn)
        tgt = self.normselfattn(tgt)
        tgt = tgt + self.crossattn(tgt, src, src, src_key_padding_mask, None)
        tgt = self.normcrossattn(tgt)
        tgt = tgt + self.ff2(self.dropout(self.ff1(tgt)))
        tgt = self.normff(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    """Implementation of the transformer decoder stack.

    Parameters
    ----------
        d_model: The dimension of decoders inputs/outputs.
        dim_feedforward: Hidden dimension of the feedforward networks.
        num_decoder_layers: Number of stacked decoders.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_decoder_layer:int ,
            nhead: int,
            dropout: float,
            normcls = nn.LayerNorm
        ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_decoder_layer):
          self.layers.append(TransformerDecoderLayer(d_model,d_ff,nhead,dropout,normcls))

    def forward(
            self,
            src: torch.FloatTensor,
            tgt: torch.FloatTensor,
            tgt_mask_attn: torch.BoolTensor,
            src_key_padding_mask: torch.BoolTensor,
            tgt_key_padding_mask: torch.BoolTensor,
        ) -> torch.FloatTensor:
        """Decodes the source sequence by sequentially passing.
        the encoded source sequence and the target sequence through the decoder stack.

        Args
        ----
            src: Batch of encoded source sentences.
                Shape of [batch_size, src_seq_len, dim_model].
            tgt: Batch of taget sentences.
                Shape of [batch_size, tgt_seq_len, dim_model].
            tgt_mask_attn: Mask to prevent attention to subsequent tokens.
                Shape of [tgt_seq_len, tgt_seq_len].
            src_key_padding_mask: Mask to prevent attention to padding in src sequence.
                Shape of [batch_size, src_seq_len].
            tgt_key_padding_mask: Mask to prevent attention to padding in tgt sequence.
                Shape of [batch_size, tgt_seq_len].

        Output
        ------
            y:  Batch of sequence of embeddings representing the predicted target tokens
                Shape of [batch_size, tgt_seq_len, dim_model].
        """
        for layer in self.layers:
          tgt = layer(src, tgt, tgt_mask_attn, src_key_padding_mask, tgt_key_padding_mask)
        return tgt

class TransformerEncoderLayer(nn.Module):
    """Single encoder layer.

    Parameters
    ----------
        d_model: The dimension of input tokens.
        dim_feedforward: Hidden dimension of the feedforward networks.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            nhead: int,
            dropout: float,
            normcls = nn.LayerNorm
        ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.nhead = nhead

        self.selfattn = MultiheadAttention(d_model, nhead, dropout)
        self.normselfattn = normcls(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.normff = normcls(d_model)

    def forward(
        self,
        src: torch.FloatTensor,
        key_padding_mask: torch.BoolTensor
        ) -> torch.FloatTensor:
        """Encodes the input. Does not attend to masked inputs.

        Args
        ----
            src: Batch of embedded source tokens.
                Shape of [batch_size, src_seq_len, dim_model].
            key_padding_mask: Mask preventing attention to padding tokens.
                Shape of [batch_size, src_seq_len].

        Output
        ------
            y: Batch of encoded source tokens.
                Shape of [batch_size, src_seq_len, dim_model].
        """
        src = src + self.selfattn(src, src, src, key_padding_mask)
        src = self.normselfattn(src)
        src = src + self.ff2(self.dropout(self.ff1(src)))
        src = self.normff(src)
        return src


class TransformerEncoder(nn.Module):
    """Implementation of the transformer encoder stack.

    Parameters
    ----------
        d_model: The dimension of encoders inputs.
        dim_feedforward: Hidden dimension of the feedforward networks.
        num_encoder_layers: Number of stacked encoders.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            dim_feedforward: int,
            num_encoder_layers: int,
            nheads: int,
            dropout: float,
            normcls = nn.LayerNorm
        ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_encoder_layers):
          self.layers.append(TransformerEncoderLayer(d_model,dim_feedforward,nheads,dropout,normcls))


    def forward(
            self,
            src: torch.FloatTensor,
            key_padding_mask: torch.BoolTensor
        ) -> torch.FloatTensor:
        """Encodes the source sequence by sequentially passing.
        the source sequence through the encoder stack.

        Args
        ----
            src: Batch of embedded source sentences.
                Shape of [batch_size, src_seq_len, dim_model].
            key_padding_mask: Mask preventing attention to padding tokens.
                Shape of [batch_size, src_seq_len].

        Output
        ------
            y: Batch of encoded source sequence.
                Shape of [batch_size, src_seq_len, dim_model].
        """
        for layer in self.layers:
          tgt = layer(src, key_padding_mask)
        return tgt


class Transformer(nn.Module):
    """Implementation of a Transformer based on the paper: https://arxiv.org/pdf/1706.03762.pdf.

    Parameters
    ----------
        d_model: The dimension of encoders/decoders inputs/ouputs.
        nhead: Number of heads for each multi-head attention.
        num_encoder_layers: Number of stacked encoders.
        num_decoder_layers: Number of stacked encoders.
        dim_feedforward: Hidden dimension of the feedforward networks.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            normcls = nn.LayerNorm
        ):
        super().__init__()
        self.enc = TransformerEncoder(d_model, dim_feedforward, num_encoder_layers, nhead,dropout,normcls)
        self.dec = TransformerDecoder(d_model, dim_feedforward, num_encoder_layers, nhead,dropout,normcls)

    def forward(
            self,
            src: torch.FloatTensor,
            tgt: torch.FloatTensor,
            tgt_mask_attn: torch.BoolTensor,
            src_key_padding_mask: torch.BoolTensor,
            tgt_key_padding_mask: torch.BoolTensor
        ) -> torch.FloatTensor:
        """Compute next token embeddings.

        Args
        ----
            src: Batch of source sequences.
                Shape of [batch_size, src_seq_len, dim_model].
            tgt: Batch of target sequences.
                Shape of [batch_size, tgt_seq_len, dim_model].
            tgt_mask_attn: Mask to prevent attention to subsequent tokens.
                Shape of [tgt_seq_len, tgt_seq_len].
            src_key_padding_mask: Mask to prevent attention to padding in src sequence.
                Shape of [batch_size, src_seq_len].
            tgt_key_padding_mask: Mask to prevent attention to padding in tgt sequence.
                Shape of [batch_size, tgt_seq_len].

        Output
        ------
            y: Next token embeddings, given the previous target tokens and the source tokens.
                Shape of [batch_size, tgt_seq_len, dim_model].
        """
        src = self.enc(src, src_key_padding_mask)
        tgt = self.dec(src, tgt, tgt_mask_attn, src_key_padding_mask, tgt_key_padding_mask)
        return tgt


class TranslationTransformer(nn.Module):
    """Basic Transformer encoder and decoder for a translation task.
    Manage the masks creation, and the token embeddings.
    Position embeddings can be learnt with a standard `nn.Embedding` layer.

    Parameters
    ----------
        n_tokens_src: Number of tokens in the source vocabulary.
        n_tokens_tgt: Number of tokens in the target vocabulary.
        n_heads: Number of heads for each multi-head attention.
        dim_embedding: Dimension size of the word embeddings (for both language).
        dim_hidden: Dimension size of the feedforward layers
            (for both the encoder and the decoder).
        n_layers: Number of layers in the encoder and decoder.
        dropout: Dropout rate.
        src_pad_idx: Source padding index value.
        tgt_pad_idx: Target padding index value.
    """
    def __init__(
            self,
            n_tokens_src: int,
            n_tokens_tgt: int,
            n_heads: int,
            dim_embedding: int,
            dim_hidden: int,
            n_layers: int,
            dropout: float,
            src_pad_idx: int,
            tgt_pad_idx: int,
            normcls = nn.LayerNorm
        ):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.srcemb = nn.Embedding(n_tokens_src, dim_embedding)
        self.tgtemb = nn.Embedding(n_tokens_tgt, dim_embedding)
        self.tf = Transformer(dim_embedding, n_heads, n_layers,n_layers,dim_hidden,dropout,normcls)
        self.Wlogits = nn.Linear(dim_embedding, n_tokens_tgt)


    def forward(
            self,
            source: torch.LongTensor,
            target: torch.LongTensor
        ) -> torch.FloatTensor:
        """Predict the target tokens logites based on the source tokens.

        Args
        ----
            source: Batch of source sentences.
                Shape of [batch_size, seq_len_src].
            target: Batch of target sentences.
                Shape of [batch_size, seq_len_tgt].

        Output
        ------
            y: Distributions over the next token for all tokens in each sentences.
                Those need to be the logits only, do not apply a softmax because
                it will be done in the loss computation for numerical stability.
                See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for more informations.
                Shape of [batch_size, seq_len_tgt, n_tokens_tgt].
        """
        src = self.srcemb(source)
        tgt = self.tgtemb(target)
        tgt = self.tf(src, tgt, self.generate_causal_mask(target), *self.generate_key_padding_mask(source,target))
        logits= self.Wlogits(tgt)
        return logits


    def generate_causal_mask(
            self,
            target: torch.LongTensor,
        ) -> tuple:
        """Generate the masks to prevent attending subsequent tokens.

        Args
        ----
            source: Batch of source sentences.
                Shape of [batch_size, seq_len_src].
            target: Batch of target sentences.
                Shape of [batch_size, seq_len_tgt].

        Output
        ------
            tgt_mask_attn: Mask to prevent attention to subsequent tokens.
                Shape of [seq_len_tgt, seq_len_tgt].

        """

        seq_len = target.shape[1]

        tgt_mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        tgt_mask = torch.triu(tgt_mask, diagonal=1).to(target.device)

        return tgt_mask

    def generate_key_padding_mask(
            self,
            source: torch.LongTensor,
            target: torch.LongTensor,
        ) -> tuple:
        """Generate the masks to prevent attending padding tokens.

        Args
        ----
            source: Batch of source sentences.
                Shape of [batch_size, seq_len_src].
            target: Batch of target sentences.
                Shape of [batch_size, seq_len_tgt].

        Output
        ------
            src_key_padding_mask: Mask to prevent attention to padding in src sequence.
                Shape of [batch_size, seq_len_src].
            tgt_key_padding_mask: Mask to prevent attention to padding in tgt sequence.
                Shape of [batch_size, seq_len_tgt].

        """

        src_key_padding_mask = source == self.src_pad_idx
        tgt_key_padding_mask = target == self.tgt_pad_idx

        return src_key_padding_mask, tgt_key_padding_mask
