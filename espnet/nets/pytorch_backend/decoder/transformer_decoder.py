#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""

from typing import Any, List, Tuple

import torch
from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import BatchScorerInterface


class DecoderLayer(torch.nn.Module):
    """Single decoder layer module.
    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
        self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
        src_attn: source attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward: feed forward layer module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = torch.nn.Linear(size + size, size)
            self.concat_linear2 = torch.nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """Compute decoded features.
        Args:
            tgt (torch.Tensor):
                decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "output_norm.", prefix + "after_norm.", state_dict)


class TransformerDecoder(BatchScorerInterface, torch.nn.Module):
    """Transfomer decoder module.

    :param int odim: output dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate for attention
    :param str or torch.nn.Module input_layer: input layer type
    :param bool use_output_layer: whether to use output layer
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(
        self,
        odim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        self_attention_dropout_rate=0.1,
        src_attention_dropout_rate=0.1,
        input_layer="embed",
        use_output_layer=True,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        layer_drop_rate=0.0,
    ):
        """Construct an Decoder object."""
        torch.nn.Module.__init__(self)
        self._register_load_state_dict_pre_hook(_pre_hook)
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer, pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        self.normalize_before = normalize_before
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
            layer_drop_rate,
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, odim)
        else:
            self.output_layer = None

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Forward decoder.
        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
                                 if input_layer == "embed"
                                 input tensor (batch, maxlen_out, #mels)
                                 in the other cases
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param torch.Tensor memory_mask: encoded memory mask,  (batch, maxlen_in)
                                         dtype=torch.uint8 in PyTorch 1.2-
                                         dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :return x: decoded token score before softmax (batch, maxlen_out, token)
                   if use_output_layer is True,
                   final block outputs (batch, maxlen_out, attention_dim)
                   in the other cases
        :rtype: torch.Tensor
        :return tgt_mask: score mask before softmax (batch, maxlen_out)
        :rtype: torch.Tensor
        """
        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, tgt_mask

    def forward_one_step(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        """Forward one step.
        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param List[torch.Tensor] cache:
            cached output list of (batch, max_time_out-1, size)
        :return y, cache: NN output value and cache per `self.decoders`.
            `y.shape` is (batch, maxlen_out, token)
        :rtype: Tuple[torch.Tensor, List[torch.Tensor]]
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    # beam search API (see ScorerInterface)
    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).
        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][l] for b in range(n_batch)])
                for l in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[l][b] for l in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
