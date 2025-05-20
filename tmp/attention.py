#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ***********************************************
#      Filename: src/sparseopt/attns/flash_attn.py
#        Author: jiff
#         Email: Jiff_Zh@163.com
#   Description: --
#        Create: 2024-12-20 13:36:43
# Last Modified: Year-month-day
# ***********************************************

import math
import torch
from torch import nn
from torch.nn import functional as F
# from transformers.models.llama.modeling_llama import repeat_kv
from transformers.models.llama.modeling_llama import repeat_kv
from tqdm import tqdm
import einops
import copy
import epsilon
import torch
from torch import Tensor
import epsilon
from typing import List, Optional


class ATTNS:

    @staticmethod
    def add_fp24(a, b):
        # a = torch.randn(32,32).to(torch.float32)
        a = (a.view(torch.int32) & torch.tensor([0xffffff00], dtype=torch.uint32).to(dtype=torch.int32,
                                                                                     device=a.device)).view(
            torch.float32)
        b = (b.view(torch.int32) & torch.tensor([0xffffff00], dtype=torch.uint32).to(dtype=torch.int32,
                                                                                     device=b.device)).view(
            torch.float32)
        c = a + b
        c = (c.view(torch.int32) & torch.tensor([0xffffff00], dtype=torch.uint32).to(dtype=torch.int32,
                                                                                     device=c.device)).view(
            torch.float32)
        return c

    @staticmethod
    def _get_attn_bias(
            slice: tuple,
            dtype: torch.dtype,
            device: torch.device,
            is_causal: bool = True,
            attention_mask: torch.Tensor = None,  # [B, K] or [B, 1, 1, K]
    ) -> torch.Tensor:
        # index of query-start, query-end, key-start, key-end
        q_s, q_e, k_s, k_e = slice
        # [1, 1, QB, KB]
        attn_bias = torch.zeros(
            1, 1, q_e - q_s, k_e - k_s,
            dtype=dtype,
            device=device
        )
        if is_causal:
            # [QB]
            q_idx = torch.arange(q_s, q_e, device=device)
            # [KB]
            k_idx = torch.arange(k_s, k_e, device=device)
            # [QB, KB]
            attn_mask = q_idx[:, None] >= k_idx[None, :]
            attn_bias.masked_fill_(
                attn_mask[None, None, :, :].logical_not(),
                torch.finfo(dtype).min
            )
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]

            # [B, 1, 1, KB] or [B, 1, QB, KB]
            if attention_mask.size(-2) == 1:  # [mask = -inf ,non-mask = 0]
                attention_mask_p = attention_mask[..., k_s: k_e].to(dtype)
            else:
                attention_mask_p = attention_mask[..., q_s: q_e, k_s: k_e]
            if attention_mask.dtype == torch.bool:
                attn_bias = attn_bias.repeat(attention_mask_p.size(0), attention_mask_p.size(1), 1, 1)  # here
                attn_bias.masked_fill_(
                    attention_mask_p.logical_not(),
                    torch.finfo(dtype).min
                )
            else:
                attn_bias = attn_bias + attention_mask_p
        return attn_bias

    @classmethod
    def eager(
            cls,
            query_states: torch.Tensor,  # [B, H, Q, D]
            key_states: torch.Tensor,  # [B, H, K, D]
            value_states: torch.Tensor,  # [B, H, K, D]
            attention_mask: torch.Tensor = None,  # [B, K]
            is_causal: bool = True,
            attention_dropout: float = 0.0,
            training: bool = False,
            softmax_dtype: torch.dtype = torch.float32,
            attn_quant=nn.Identity()
    ) -> torch.Tensor:
        # repeat kv
        key_states = repeat_kv(
            key_states, query_states.size(1) // key_states.size(1)
        )
        value_states = repeat_kv(
            value_states, query_states.size(1) // value_states.size(1)
        )

        K, Q = key_states.size(-2), query_states.size(-2)

        # [B, 1, Q, K]
        attn_bias = cls._get_attn_bias(
            slice=(0, Q, 0, K),
            dtype=query_states.dtype,
            device=query_states.device,
            is_causal=is_causal and Q > 1,
            attention_mask=attention_mask,
        )

        # [B, H, Q, K]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) \
                       / math.sqrt(query_states.size(-1))
        attn_weights = attn_weights + attn_bias

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=softmax_dtype).to(query_states.dtype)
        # attn_weights = attn_weights.to(softmax_dtype)
        # attn_weights -= attn_weights.max(-1, keepdim=True).values
        # attn_weights = attn_weights.exp()
        # attn_weights /= attn_weights.sum(-1, keepdim=True)
        # attn_weights = attn_weights.to(query_states)

        attn_weights = attn_quant(attn_weights)  # prefill_quant

        attn_weights = nn.functional.dropout(attn_weights, p=attention_dropout, training=training)

        # [B, H, Q, D]
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output

    @classmethod
    def sdpa(
            cls,
            query_states: torch.Tensor,  # [B, H, Q, D]
            key_states: torch.Tensor,  # [B, H, K, D]
            value_states: torch.Tensor,  # [B, H, K, D]
            attention_mask: torch.Tensor = None,  # [B, K]
            is_causal: bool = True,
            attention_dropout: float = 0.0,
            training: bool = False,
    ) -> torch.Tensor:
        # repeat kv
        key_states = repeat_kv(
            key_states, query_states.size(1) // key_states.size(1)
        )
        value_states = repeat_kv(
            value_states, query_states.size(1) // value_states.size(1)
        )
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        return F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            is_causal=is_causal,
            dropout_p=attention_dropout if training else 0.0,
            attn_mask=attention_mask,
        )

    @classmethod
    def flash(
            cls,
            query_states: torch.Tensor,  # [B, H, Q, D]
            key_states: torch.Tensor,  # [B, H, K, D]
            value_states: torch.Tensor,  # [B, H, K, D]
            block_size: tuple = (8192, 64),  # [1024 ,1024]
            softmax_dtype: torch.dtype = torch.float32,
            is_causal: bool = True,
            attention_mask: torch.Tensor = None,  # [B, K]
            attention_dropout: float = 0.0,
            training: bool = False,
            safe_mem: bool = False,
            verbose: bool = False,
            attn_quant=nn.Identity(),
            flash_attention_method='flash_attention_eplison',
            matmul_type='bf16_matmul'
    ) -> torch.Tensor:

        # print('query_states',query_states.size())
        # print('key_states',key_states.size())
        # print('value_states',value_states.size())

        func = cls._flash_core_eplison

        if isinstance(block_size, int):
            block_size = (block_size, block_size)
        n_row, n_col = block_size

        # repeat kv
        key_states = repeat_kv(
            key_states, query_states.size(1) // key_states.size(1)
        )
        value_states = repeat_kv(
            value_states, query_states.size(1) // value_states.size(1)
        )

        B, H, Q, _ = query_states.size()
        _, _, K, D = value_states.size()

        attn_output = torch.zeros(  # 声明res，类型为value.dtype
            B, H, Q, D,
            dtype=value_states.dtype,
            device=value_states.device,
        )
        # print('attn_output',attn_output.size())

        # torch.cuda.empty_cache()
        for i in tqdm(
                range(0, Q, n_row),
                desc="Flash Attention Query Loop",
                position=0,
                # leave=True,
                disable=not verbose,
        ):
            func(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attn_output=attn_output,
                slice=(i, min(i + n_row, Q)),
                n_col=n_col,
                softmax_dtype=softmax_dtype,
                is_causal=is_causal and Q > 1,
                attention_mask=attention_mask,
                attention_dropout=attention_dropout,
                training=training,
                safe_mem=safe_mem,
                verbose=verbose,
                attn_quant=attn_quant,
                matmul_type=matmul_type

            )
            # print('loop attn_output',attn_output.size())

        # print('final attn_output',attn_output.size())

        return attn_output

    @classmethod
    def _flash_core_eplison(
            cls,
            query_states: torch.Tensor,  # [B, H, Q, D]
            key_states: torch.Tensor,  # [B, H, K, D]
            value_states: torch.Tensor,  # [B, H, K, D]
            attn_output: torch.Tensor,  # [B, H, Q, D]
            slice: tuple,
            n_col: int = 1024,
            softmax_dtype: torch.dtype = torch.float32,
            is_causal: bool = True,
            attention_mask: torch.Tensor = None,  # [B, K]
            attention_dropout: float = 0.0,
            training: bool = False,
            safe_mem: bool = False,
            verbose: bool = False,
            attn_quant=nn.Identity,
            matmul_type: str = ''
    ):

        inputs_dtype = query_states.dtype

        B, H, Q, _ = query_states.size()
        _, _, K, D = value_states.size()
        q_s, q_e = slice  # 改切片的开始和结束
        # print('slice',slice)
        n_block = math.ceil(K / n_col)  # the number of block

        # print('attention_mask',attention_mask.size())
        # [B, H, N_ROW, 1, N_BLOCK]
        # biggest product of `q @ k` seen
        m_buffer = torch.zeros(
            B, H, q_e - q_s, 1, n_block,
            dtype=softmax_dtype,
            device=query_states.device,
        )  # half
        m_buffer.fill_(torch.finfo(m_buffer.dtype).min)
        # sum of the `exp(q @ k - m_buffer)` seen
        l_buffer = torch.zeros_like(m_buffer).to(softmax_dtype)  # fp32, the per-block sum of exp(input - max)

        # [B, H, N_ROW, D, N_BLOCK]
        output_buffer = torch.zeros(
            B, H, q_e - q_s, D, n_block,
            dtype=softmax_dtype,
            device=query_states.device,
        )  # fp32, the per-block result of "A @ V"
        # print('output_buffer',output_buffer.size())

        for j in tqdm(
                range(0, K, n_col),
                desc="Flash Attention Key Loop",
                position=1,
                leave=True,
                disable=not verbose,
        ):

            if torch.cuda.is_available() and safe_mem:
                torch.cuda.empty_cache()
            bk = j // n_col  # block id

            ## 1. Q @ K
            # [B, 1, N_ROW, N_COL]
            attn_bias = cls._get_attn_bias(
                (q_s, q_e, j, min(K, j + n_col)),
                dtype=query_states.dtype,
                device=query_states.device,
                is_causal=is_causal,
                attention_mask=attention_mask,
            )
            # bf16_matmul epsilon fp32_to_int8_triton mul_add_split
            # [B, H, N_ROW, N_COL]

            attn_weights = torch.matmul(
                query_states[:, :, q_s: q_e, :],
                key_states[:, :, j: j + n_col, :].transpose(2, 3)
            ) / math.sqrt(query_states.size(-1))

            attn_weights = cls.add_fp24(
                attn_weights.to(softmax_dtype),
                attn_bias.to(softmax_dtype)
            ).to(attn_weights.dtype)  # A @ V + bias fp_24_add

            attn_weights, local_sum, local_max = epsilon.nn.softmax(attn_weights.float(), return_factor=True,
                                                                    custom_ops=False)  # bf16

            # attn_weights = attn_weights.to(torch.bfloat16)
            # local_sum = attn_weights.to(torch.bfloat16)
            # local_max = attn_weights.to(torch.bfloat16)

            m_local = local_max
            if bk == 0:
                m_buffer[..., bk] = m_local
                l_buffer[..., bk] = local_sum.to(softmax_dtype)
            else:
                m_buffer[..., bk] = torch.max(m_buffer[..., bk - 1], m_local)  # 截至当前块的最大值
                torch_exp = torch.exp(m_buffer[..., bk - 1] - m_buffer[..., bk]).to(softmax_dtype)  # 更新的exp系数 fp32
                torch_exp_local = torch.exp(m_local - m_buffer[..., bk]).to(softmax_dtype)  # 更新的exp系数 fp32
                l_buffer[..., bk] = torch_exp * l_buffer[..., bk - 1] + torch_exp_local * local_sum.to(softmax_dtype)

            attn_weights = nn.functional.dropout(
                attn_weights, p=attention_dropout, training=training
            )
            attn_weights = attn_quant(attn_weights)  # prefill_quant
            # [B, H, N_ROW, D]
            output_buffer[..., bk] = torch.matmul(
                attn_weights.to(torch.bfloat16), value_states[:, :, j: j + n_col, :]
            ).to(softmax_dtype)

            if bk >= 1:
                output_buffer[..., bk] = (
                        torch_exp * l_buffer[..., bk - 1] * output_buffer[..., bk - 1]  # fp32 mul
                        +
                        torch_exp_local * output_buffer[..., bk]
                )

            output_buffer[..., bk] = (output_buffer[..., bk] / l_buffer[..., bk]).to(softmax_dtype)  # error 在这

        attn_output[:, :, q_s: q_e, :] = output_buffer[..., -1].to(inputs_dtype)


def compare(x1: torch.Tensor, x2: torch.Tensor, prompt: str):
    diff = (x1 - x2).abs()
    sim = torch.cosine_similarity(x1, x2, dim=-1).mean()
    print(
        f'{prompt}:\n'
        f'\tmax: {diff.max().item():.5e}\n'
        f'\tmin: {diff.min().item():.5e}\n'
        f'\tmean: {diff.mean().item():.5e}\n'
        f'\tsim: {sim.item():.5e}'
    )


if __name__ == "__main__":
    verbose = True
    # verbose = False

    is_causal = False
    is_causal = True

    block_size = 128
    # block_size = 8192
    # block_size = (1024, 8192 * 8)

    seq_len = 1024
    # seq_len = 64 * 1024
    # seq_len = 128 * 1024
    # seq_len = 256 * 1024
    B, KH, QH, S, D = 2, 4, 32, seq_len, 128

    dtype = torch.bfloat16
    # dtype = torch.float
    # dtype = torch.float64

    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    # else:
    device = torch.device('cpu')

    query_states = torch.randn(B, QH, S, D, dtype=dtype, device=device)
    key_states = torch.randn(B, KH, S, D, dtype=dtype, device=device)
    value_states = torch.randn(B, KH, S, D, dtype=dtype, device=device)
    attention_mask = torch.randn(B, S, device=device) > 0
    # attention_mask = torch.arange(S, device=device)[None, :] > torch.randint(0, S, (B, 1))
    attention_mask = (~attention_mask).to(dtype) * torch.finfo(dtype).min
    # attention_mask = None

    # Flash
    print("### Running Flash ... ###")
    import time

    st_time = time.time()
    flash_output = ATTNS.flash(
        query_states, key_states, value_states,
        is_causal=is_causal, attention_mask=attention_mask,
        block_size=block_size,
        verbose=verbose,
    )
    print(f"### Time used: {time.time() - st_time:.3f} ###")
    print(f'flash\t{flash_output.shape}\t{flash_output.mean().item()}')

    ## SDPA
    print("### Running SDPA ... ###")
    st_time = time.time()
    sdpa_output = ATTNS.sdpa(
        query_states, key_states, value_states,
        is_causal=is_causal, attention_mask=attention_mask
    )
    print(f"### Time used: {time.time() - st_time:.3f} ###")
    print(f'sdpa\t{sdpa_output.shape}\t{sdpa_output.mean().item()}')

    ## Eager
    try:
        print("### Running Eager ... ###")
        st_time = time.time()
        eager_output = ATTNS.eager(
            query_states, key_states, value_states,
            is_causal=is_causal, attention_mask=attention_mask
        )
        print(f"### Time used: {time.time() - st_time:.3f} ###")
        print(f'eager\t{eager_output.shape}\t{eager_output.mean().item()}')

        compare(flash_output, eager_output, 'flash vs eager')
        compare(sdpa_output, eager_output, 'sdpa vs eager')

    except torch.cuda.OutOfMemoryError:
        print('eager OOM')

    compare(flash_output, sdpa_output, 'flash vs sdpa')

