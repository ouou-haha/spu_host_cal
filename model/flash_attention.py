import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math
from tqdm import tqdm
import method
from torch import nn


def flash_attention(
        query_states: torch.Tensor,  # [B, H, Q, D]
        key_states: torch.Tensor,  # [B, H, K, D]
        value_states: torch.Tensor,  # [B, H, K, D]
        attn_output: torch.Tensor,  # [B, H, Q, D]
        block_size: tuple = (8192, 64),  # Q = 8192 , seq(KV) = 64
        n_col: int = 64,
        softmax_dtype: torch.dtype = torch.float32,
        attention_mask: torch.Tensor = None,
        attn_quant=nn.Identity(),
):
    global torch_exp, torch_exp_local
    inputs_dtype = query_states.dtype
    B, H, Q, _ = query_states.size()
    _, _, K, D = value_states.size()
    n_block = math.ceil(K / n_col)  # the number of block

    # initial m and d buffer
    m_buffer = torch.zeros(B, H, Q, 1, n_block, dtype=softmax_dtype)  # buffer of max(input)
    d_buffer = torch.zeros_like(m_buffer).to(softmax_dtype)  # fp32, buffer of sum of exp(input - max(input))
    output_buffer = torch.zeros(B, H, Q, D, n_block, dtype=softmax_dtype)  # A @ V

    m_buffer.fill_(torch.finfo(m_buffer.dtype).min)  # initial m_buffer with minimum bf16 value

    for j in tqdm(
            range(0, K, n_col),
            desc="Flash Attention Query Loop",
            position=0,
            disable=True,
    ):
        bank_num = j // n_col

        # [B, 1, N_ROW, N_COL]
        attn_bias = method.get_attn_bias(
            (0, Q, j, min(K, j + n_col)),
            dtype=query_states.dtype,
            device=query_states.device,
            is_causal=True,
            attention_mask=attention_mask,
        )

        # Q @ K    [B, H, Q, D] @ [B, H, K, D] -> [B, H, Q, K]
        attn_weight = torch.matmul(query_states, key_states[:, :, j: j + n_col, :].transpose(2, 3))
        attn_weight /= math.sqrt(query_states.size(-1))

        attn_weight = method.add_fp24(
            attn_weight.to(softmax_dtype),
            attn_bias.to(softmax_dtype)
        ).to(attn_weight.dtype)  # A @ V + bias fp_24_add

        # exp(ai - max) sum(), m
        attn_weight, local_sum, local_max = method.softmax(attn_weight.float())

        m_local = local_max
        # 更新历史最大值与系数
        if bank_num == 0:
            m_buffer[..., bank_num] = m_local
            d_buffer[..., bank_num] = local_sum.to(softmax_dtype)
        else:
            m_buffer[..., bank_num] = torch.max(m_buffer[..., bank_num - 1], m_local)
            torch_exp = torch.exp(m_buffer[..., bank_num - 1] - m_buffer[..., bank_num]).to(softmax_dtype)
            torch_exp_local = torch.exp(m_local - m_buffer[..., bank_num]).to(softmax_dtype)
            d_buffer[..., bank_num] = d_buffer[..., bank_num - 1] * torch_exp + torch_exp_local * local_sum

        attn_weight = attn_quant(attn_weight)  # prefill_quant

        # attn_weight @ V    [B, H, Q, K] @ [B, H, K, D] -> [B, H, Q, D]
        output_buffer[..., bank_num] = torch.matmul(
            attn_weight.to(torch.bfloat16), value_states[:, :, j: j + n_col, :]
        ).to(softmax_dtype)

        if bank_num >= 1:
            output_buffer[..., bank_num] = (
                    torch_exp * d_buffer[..., bank_num - 1] * output_buffer[..., bank_num - 1]  # fp32 mul
                    +
                    torch_exp_local * output_buffer[..., bank_num]
            )

        output_buffer[..., bank_num] = (output_buffer[..., bank_num] / d_buffer[..., bank_num]).to(
            softmax_dtype)  # error 在这

    attn_output[:, :, :, :] = output_buffer[..., -1].to(inputs_dtype)


if __name__ == "__main__":
    pass
