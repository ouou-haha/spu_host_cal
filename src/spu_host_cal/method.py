import torch

_SOFTMAX_GROUP = 64


def softmax(
        input: torch.Tensor, debug: bool = False, custom_ops: bool = False, return_factor=True
):
    """
    Computes the softmax of each element in the input tensor.

    Args:
        input (torch.Tensor): The input tensor.
        dim (int, optional): The dimension to reduce. Default is -1.
        debug (bool, optional): If True, print debug information. Default is False.

    Returns:
        torch.Tensor: The result tensor with the softmax of each element.
    """
    if debug:
        print(f"Original input shape: {input.shape}")

    # Save shape and view the input tensor as a 3D tensor
    shape = input.size()  # [16, 32, 1403, 64]
    # print('input size',shape)
    input = input.view(-1, input.size(-1) // _SOFTMAX_GROUP, _SOFTMAX_GROUP)  # [S,num_bank,bank]
    input = torch.max(
        input, torch.tensor(-10000.0, device=input.device, dtype=input.dtype)
    )  # 截断

    if debug:
        print(f"Shape after view to 3D tensor: {input.shape}")

    # Compute max of each group
    local_max = input.max(-1, keepdim=False).values  # channel分成若干bank，每个bank的最大值 [S,num_bank]
    if debug:
        print(f"Shape of local_max: {local_max.shape}")

    d = torch.zeros(input.shape[:-2], device=input.device, dtype=input.dtype)
    mj = torch.full(
        input.shape[:-2], float("-inf"), device=input.device, dtype=input.dtype
    )
    mi = []
    unorm = []
    ipts = []

    if debug:
        print(f"Shape of d: {d.shape}")
        print(f"Shape of mj: {mj.shape}")

    # Loop through -2 dimension
    for i in range(input.size(-2)):  # for num_bank
        mj_last = mj
        mj = torch.maximum(local_max[..., i], mj)  # 截至当前块的最大值
        mi.append(mj)

        ipt = input[..., i, :] - mj.unsqueeze(-1)  # 输入 - 当前块的最大值
        ipts.append(ipt.flatten())

        input_exp = torch.exp(input[..., i, :] - mj.unsqueeze(-1))

        local_sum = input_exp.sum(-1, keepdim=False)  # 当前sum(exp(i - max))
        unorm.append(input_exp)  # 每个bank的exp(i - max)

        torch_exp = torch.exp(mj_last - mj).squeeze()
        d = d * torch_exp + local_sum  # 截至目前sum和
        # print('d',d.reshape(-1)[:256].tolist())
        # print(i)

    if debug:
        print(f"Shape of torch_exp after loop: {torch_exp.shape}")
        print(f"Shape of local_sum after loop: {local_sum.shape}")
        print(f"Shape of mj after loop: {mj.shape}")
        print(f"Shape of d after loop: {d.shape}")

    mi = torch.stack(mi, dim=-1)
    if debug:
        print(f"Shape of mi after stack: {mi.shape}")

    global_max = mi.max(-1, keepdim=False).values  # 全局最大值

    denominator = 1 / d
    denominator = denominator.unsqueeze(-1).unsqueeze(-1)  # 全局sum和

    if debug:
        print(f"Shape of global_max: {global_max.shape}")
        print(f"Shape of denominator: {denominator.shape}")

    # Scale factor

    scale_factor = torch.exp(mi - global_max.unsqueeze(-1)).unsqueeze(-1)
    unorm = torch.stack(unorm, dim=-1).transpose(-1, -2)  # 每块exp(i - max)

    if debug:
        print(f"Shape of scale_factor: {scale_factor.shape}")
        print(f"Shape of unorm: {unorm.shape}")

    result = scale_factor * unorm
    # result = result * denominator

    if debug:
        print(f"Shape of result before reshape: {result.shape} -> {shape}")

    # Remove padded values
    result = result.reshape(shape)

    if debug:
        print(f"Final result shape: {result.shape}")

    # 全局的和 d
    # 全局最大值 global_max
    if return_factor:
        # print('r',d.reshape(-1)[:256].tolist())

        return result, d.view(*shape[:-1], 1), global_max.view(*shape[:-1], 1)
    else:
        return result


def get_attn_bias(
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


if __name__ == '__main__':
    pass
