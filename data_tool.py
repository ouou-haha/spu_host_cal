# !/usr/bin/env python3
import argparse
import numpy as np
import os
import torch
from typing import Tuple, Dict, Any
import torch.nn.functional as F

print(f"torch version: {torch.__version__} ")
print(f"data_tool version: {5.9}")

INT4: str = "int4"
INT8: str = "int8"
BF16: str = "bf16"
UINT64: str = "uint64"
FP8E5M2: str = "fp8_e5m2"
FP8E4M3: str = "fp8_e4m3"
FP32: str = "fp32"
INT32: str = "int32"
UINT32: str = "uint32"
NA: str = "na"

dtype_torch_map = {
    "int8": torch.int8,
    "bf16": torch.bfloat16,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def load_bin_tensor(path: str, dtype: str):
    if dtype == BF16:
        return torch.from_numpy(np.fromfile(path, dtype=np.uint16)).view(torch.bfloat16)
    elif dtype == INT8:
        return torch.from_numpy(np.fromfile(path, dtype=np.int8)).view(torch.int8)
    elif dtype == FP32:
        return torch.from_numpy(np.fromfile(path, dtype=np.float32))
    elif dtype == INT32:
        return torch.from_numpy(np.fromfile(path, dtype=np.int32))
    elif dtype == INT4:
        return load_int4_from_bin(path)


def save_tensor_bin(tensor: torch.Tensor, path: str, dtype: str = None):
    if dtype is None:
        if tensor.dtype == torch.bfloat16:
            tensor.view(torch.float16).numpy().tofile(path)
        elif tensor.dtype in (torch.float32, torch.int8):
            tensor.numpy().tofile(path)
    else:
        if dtype.lower() == INT4:
            tensor = tensor.to(torch.int8)
            save_int4_as_bin(tensor, path)
        elif dtype.lower() == BF16:
            tensor.view(torch.float16).numpy().tofile(path)
        elif dtype.lower() in (FP32, INT8):
            tensor.numpy().tofile(path)
    return True


def sim(A, B):
    A_flat = A.flatten().to(torch.float32)
    B_flat = B.flatten().to(torch.float32)
    close = torch.isclose(A_flat, B_flat)
    # print(f"Compare {len(A_flat)}\n");

    return F.cosine_similarity(A_flat, B_flat, dim=0).item()


def sim_bin(bin_file_path1: str, bin_file_path2: str, dtype: str):
    if not os.path.exists(bin_file_path1):
        raise FileNotFoundError(f"File not found: {bin_file_path1}")
    if not os.path.exists(bin_file_path2):
        raise FileNotFoundError(f"File not found: {bin_file_path2}")

    if dtype == BF16:
        A = torch.from_numpy(np.fromfile(bin_file_path1, dtype=np.float16)).view(torch.bfloat16)
        B = torch.from_numpy(np.fromfile(bin_file_path2, dtype=np.float16)).view(torch.bfloat16)
    else:
        raise ValueError(f"unsupported sim type")

    A_flat = A.flatten().to(torch.float32)
    B_flat = B.flatten().to(torch.float32)
    close = torch.isclose(A_flat, B_flat)
    print(f"Number of matching elements: {close.sum().item()} / {close.numel()} \n"
          f"sim:{F.cosine_similarity(A_flat, B_flat, dim=0).item()}")

    return close.all().item()


def dma_format_convert(w, c, tensor, bank_size, forward):
    gc = int(c / bank_size)
    cg = bank_size
    if forward == 1:
        return tensor.reshape(w, c).reshape(w, gc, cg).permute(1, 0, 2)
    elif forward == 3:
        return tensor.reshape(w, c)
    else:
        return tensor.reshape(gc, w, cg).permute(1, 0, 2).reshape(w, c)


def load_int4_from_bin(filename):
    with open(filename, 'rb') as f:
        byte_data = f.read()

    int4_list = []

    for byte in byte_data:
        first = byte & 0xF
        second = (byte >> 4) & 0xF

        def int4_sign_extend(val):
            return val - 16 if val >= 8 else val

        int4_list.append(int4_sign_extend(first))
        int4_list.append(int4_sign_extend(second))

    return torch.tensor(int4_list, dtype=torch.int8)


def save_int4_as_bin(tensor, filename):
    byte_array = bytearray()

    tensor_flat = tensor.flatten()
    if torch.any((tensor_flat < -8) | (tensor_flat > 7)):
        raise ValueError("value in Tensor must between -8 and 7")

    for i in range(0, tensor_flat.numel(), 2):
        first_int4 = tensor_flat[i].item()
        second_int4 = tensor_flat[i + 1].item()

        byte_value = (second_int4 & 0xF) << 4 | (first_int4 & 0xF)
        byte_array.append(byte_value)

    with open(filename, 'wb') as f:
        f.write(bytes(byte_array))


def float32_to_bf24_as_float32(arr: np.ndarray) -> np.ndarray:
    assert arr.dtype == np.float32
    bits = arr.view(np.uint32)
    bits_bf24 = bits & 0xFFFFFF00
    return bits_bf24.view(np.float32)


def generate_matrix(w: int, c: int, datatype: torch.dtype) -> torch.Tensor:
    if datatype == torch.int8:
        return torch.randint(-10, 10, (w, c), dtype=torch.int8)
    else:
        return torch.randn(w, c).to(datatype)


def save_tensor_as_decimal_txt(tensor: torch.Tensor, txt_file: str) -> None:
    dtype = tensor.dtype
    tensor = tensor.reshape(1, -1)
    with open(txt_file, 'w') as f:
        for row in tensor:
            if dtype == torch.int8:
                line = " ".join([str(int(x)) for x in row])
            else:
                line = " ".join([str(float(x)) for x in row])
            f.write(line + " ")


def get_topk_index(bank_vec: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    abs_bank_vec = torch.abs(bank_vec)
    indices = torch.arange(len(bank_vec))
    combined = list(zip(abs_bank_vec.tolist(), indices.tolist()))
    combined_sorted = sorted(combined, key=lambda bank_vec: (-bank_vec[0], bank_vec[1]))

    topk_indices = torch.tensor([idx for _, idx in combined_sorted[:k]])
    sorted_topk_indices, _ = torch.sort(topk_indices)
    return sorted_topk_indices, topk_indices, bank_vec[topk_indices]


def bank_sparse(block: torch.Tensor, nnz: int) -> Dict[str, Any]:
    block = block.to(torch.float32)
    sorted_topk_indices, topk_indices, _ = get_topk_index(block, nnz)
    _, topk2_indices, _ = get_topk_index(block, nnz + 1)
    all_indices = torch.arange(block.numel())
    lp_indices = torch.tensor([i for i in all_indices if i not in topk_indices])

    hp_block = block[sorted_topk_indices]
    lp_block = block[lp_indices]
    hp_block_zeros = block.clone()
    hp_block_zeros[lp_indices] = 0
    lp_block_zeros = block.clone()
    lp_block_zeros[sorted_topk_indices] = 0
    bm = 0
    for idx in sorted_topk_indices.tolist():
        bm |= (1 << idx)

    return {
        'topk_indices': topk_indices,
        'sorted_topk_indices': sorted_topk_indices,
        'lp_indices': lp_indices,
        'hp_block': hp_block,
        'hp_block_zeros': hp_block_zeros,
        'lp_block_zeros': lp_block_zeros,
        'lp_block': lp_block,
        'bitmask': bm
    }


def bank_quantize(block: torch.Tensor, in_dtype: torch.dtype, out_dtype: str):
    if out_dtype == BF16:
        return {
            'scale': torch.tensor([1.0]).to(torch.float32),
            'qnt_block': block.clone().to(torch.bfloat16),
        }
    block = block.to(torch.float32)
    max_abs = torch.max(torch.abs(block))

    if out_dtype == FP8E5M2:
        scale = 57344.0 / max_abs
    elif out_dtype == FP8E4M3:
        scale = 448.0 / max_abs
    elif out_dtype == INT8:
        scale = 127.0 / max_abs
    elif out_dtype == INT4:
        scale = 7.0 / max_abs
    else:
        scale = 1.0

    if out_dtype == INT8:
        q_min, q_max = -127, 127
        scaled = block * scale
        qnt_block = torch.round(scaled).clamp(q_min, q_max).to(torch.int8)
    elif out_dtype == INT4:
        q_min, q_max = -7, 7
        scaled = block * scale
        qnt_block = torch.round(scaled).clamp(q_min, q_max).to(torch.int8)
    elif out_dtype == FP8E5M2:
        q_min, q_max = -57344, 57344
        scaled = block * scale
        qnt_block = torch.round(scaled).clamp(q_min, q_max).to(torch.float8_e5m2)
    elif out_dtype == FP8E4M3:
        q_min, q_max = -448, 448
        scaled = block * scale
        qnt_block = torch.round(scaled).clamp(q_min, q_max).to(torch.float8_e4m3fn)
    else:
        raise ValueError(f"Unsupported out_dtype: {out_dtype}")

    dnt_scale = torch.tensor([1.0 / scale]) if scale != 0 else torch.tensor([0.0])
    return {
        'scale': dnt_scale,
        'qnt_block': qnt_block
    }


def gen_data_topk(w: int,
                  c: int,
                  k: int,
                  idtype: str,
                  ) -> Dict[str, Any]:
    input_tensor = generate_matrix(w, c, dtype_torch_map[idtype])
    output_tensor = torch.zeros(w, k, dtype=input_tensor.dtype)
    index = torch.zeros(w, k, dtype=torch.int32)
    for i in range(w):
        block = input_tensor[i].clone()
        _, block_index, block_topk = get_topk_index(block, k)
        output_tensor[i] = block_topk
        index[i] = block_index

    return {
        'input_tensor': input_tensor,
        'output_tensor': torch.abs(output_tensor),
        'index': index,
    }


def gen_data_ds2qnt(
        w: int,
        c: int,
        nnz: int,
        idtype: str,
        odtype: str,
        bank_size: int = 64,
):
    bank_num = int(c / bank_size)
    input_tensor = generate_matrix(w, c, dtype_torch_map[idtype])
    if odtype == INT4:
        input_sparse_qnt = torch.zeros(w, bank_num, nnz, dtype=torch.int8)
    else:
        input_sparse_qnt = torch.zeros(w, bank_num, nnz, dtype=dtype_torch_map[odtype])
    scale = torch.zeros(w, bank_num)
    bitmasks = np.zeros((w, bank_num), dtype=np.uint64)
    index = torch.zeros(w, bank_num, nnz, dtype=torch.int8)
    if bank_size == 32:
        bitmasks = np.zeros((w, bank_num), dtype=np.uint32)
    for i in range(w):
        for j in range(bank_num):
            block = input_tensor[i][j * bank_size: (j + 1) * bank_size].clone()
            block_nnz = bank_sparse(block, nnz)
            block_nnz_qnt = bank_quantize(block_nnz['hp_block'], torch.bfloat16, odtype)
            input_sparse_qnt[i][j] = block_nnz_qnt['qnt_block']
            scale[i][j] = block_nnz_qnt['scale']
            bitmasks[i][j] = block_nnz['bitmask']
            index[i][j] = block_nnz['sorted_topk_indices']

    return {
        'input_tensor': input_tensor,
        'output_tensor': input_sparse_qnt,
        'scale': scale,
        'bitmasks': bitmasks,
        'index': index,
    }


def gen_data_s2ddqnt(
        w: int,
        c: int,
        nnz: int,
        idtype: str,
        odtype: str,
) -> Dict[str, Any]:
    data_ds2qnt = gen_data_ds2qnt(w, c, nnz, odtype, idtype)

    bank_num = c // 64

    input_sparse_qnt = data_ds2qnt['output_tensor'].reshape(w, bank_num, nnz)  # [w, bank_num, nnz]
    input_scale = data_ds2qnt['scale'].reshape(w, bank_num)  # [w, bank_num]
    input_index = data_ds2qnt['index'].reshape(w, bank_num, nnz)  # [w, bank_num, nnz]

    output_dense_qnt = torch.zeros(w, bank_num, 64, dtype=torch.float32)
    batch_idx = torch.arange(w).view(w, 1, 1).expand(w, bank_num, nnz)
    bank_idx = torch.arange(bank_num).view(1, bank_num, 1).expand(w, bank_num, nnz)
    value_idx = input_index.to(torch.long)  # [w, bank_num, nnz]

    output_dense_qnt[batch_idx, bank_idx, value_idx] = input_sparse_qnt.to(torch.float32)
    output_dense_dqnt = output_dense_qnt * input_scale.unsqueeze(-1).to(torch.float32)
    output_dense_dqnt = output_dense_dqnt.reshape(w, c).to(dtype_torch_map[odtype])

    return {
        'input_sparse_qnt': input_sparse_qnt,
        'input_index': input_index,
        'input_scale': input_scale,
        'output_dense_qnt': output_dense_qnt,
        'output_dense_dqnt': output_dense_dqnt,
        'origin_tensor': data_ds2qnt['input_tensor'],
    }


def gen_data_sparse_mask(
        w: int = 64,
        k: int = 64,
        c: int = 64,
        nnz: int = 32,
        bank_size: int = 64,
        input_dtype: torch.dtype = torch.bfloat16,
        weight_dtype: torch.dtype = torch.bfloat16,
        out_dtype: torch.dtype = torch.bfloat16
) -> Dict[str, torch.Tensor]:
    if nnz not in (8, 16, 32, 64):
        raise ValueError(f"unsupported nnz")
    nnz = int(nnz)
    bank_num = int(c / bank_size)
    input_matrix = generate_matrix(w, c, input_dtype)

    input_dense = input_matrix
    input_sparse = torch.zeros_like(input_dense)
    input_sparse_nnz = torch.zeros((w, bank_num * nnz), dtype=input_dense.dtype)
    if c > 32:
        bitmasks = np.zeros((w, bank_num), dtype=np.uint64)
    else:
        bitmasks = np.zeros((w, bank_num), dtype=np.uint32)

    weight_dense = generate_matrix(k, c, weight_dtype)

    if nnz > c:
        raise ValueError(f"nnz > c: {nnz}")

    for i in range(w):
        for j in range(bank_num):
            block = input_dense[i][j * bank_size: (j + 1) * bank_size].to(torch.float32)
            result = bank_sparse(block, nnz)
            bitmasks[i][j] = result['bitmask']
            input_sparse[i][j * bank_size: (j + 1) * bank_size] = result['hp_block_zeros']
            input_sparse_nnz[i][j * nnz: (j + 1) * nnz] = result['hp_block']

    output = input_sparse.to(torch.float32) @ weight_dense.to(torch.float32).T
    output = output.to(out_dtype)

    if c > 64:
        weight_dense = dma_format_convert(k, c, weight_dense, bank_size, 1)  # format convert for wangwen
        bitmasks = bitmasks.flatten(order='F')

    return {
        'input_dense': input_matrix,
        'output': output,
        'input_sparse_nnz': input_sparse_nnz,
        'weight_dense': weight_dense,
        'bitmasks': bitmasks
    }


def gen_data_sparse_hp_lp(
        w: int = 64,
        k: int = 64,
        c: int = 64,
        nnz: int = 32,
        bank_size: int = 64,
        hp_dtype: str = INT8,
        lp_dtype: str = INT4,
        weight_dtype: str = INT8,
        out_dtype: str = BF16
) -> Dict[str, Any]:
    if nnz not in (4, 8, 16, 32, 64):
        raise ValueError(f"unsupported nnz")
    nnz = int(nnz)
    if c < 64:
        bank_size = c
    bank_num = int(c / bank_size)
    input_matrix = generate_matrix(w, c, torch.bfloat16)
    hp_tensor = torch.zeros(w, nnz * bank_num, dtype=dtype_torch_map.get(hp_dtype))  # output
    hp_tensor_zeros = torch.zeros(w, c, dtype=dtype_torch_map.get(hp_dtype))
    lp_tensor_encoded = torch.zeros(w, c, dtype=torch.int8)  # int4 store in int8 temply
    lp_tensor_zeros = torch.zeros(w, c, dtype=torch.int8)  # int4 store in int8 temply
    hp_scale = torch.zeros(w, bank_num, dtype=torch.float32)
    lp_scale = torch.zeros(w, bank_num, dtype=torch.float32)
    bitmasks = np.zeros((w, bank_num), dtype=np.uint64)
    weight_tensor = torch.randn(k, c, dtype=torch.bfloat16)
    weight_qnt = torch.zeros(k, c, dtype=dtype_torch_map.get(weight_dtype))
    weight_scale = torch.zeros(k, bank_num, dtype=torch.float32)
    sort_index = torch.zeros(w, bank_num, nnz, dtype=torch.int8)

    for i in range(w):
        for j in range(bank_num):
            block = input_matrix[i][j * bank_size: (j + 1) * bank_size].to(torch.float32)
            sparse_res = bank_sparse(block, nnz)
            hp_sparse = sparse_res['hp_block']
            lp_sparse = sparse_res['lp_block']
            bm = sparse_res['bitmask']
            sort_index[i][j] = sparse_res['sorted_topk_indices']

            hp_sparse_qnt = bank_quantize(hp_sparse, torch.bfloat16, hp_dtype)
            lp_sparse_qnt = bank_quantize(lp_sparse, torch.bfloat16, lp_dtype)
            scale_hp_fp24 = torch.from_numpy(float32_to_bf24_as_float32(hp_sparse_qnt['scale'].numpy()))
            scale_lp_fp24 = torch.from_numpy(float32_to_bf24_as_float32(lp_sparse_qnt['scale'].numpy()))

            lp_tensor_encoded[i][j * bank_size: (j + 1) * bank_size][sparse_res['lp_indices']] = lp_sparse_qnt[
                'qnt_block']
            if lp_dtype == INT4:
                lp_tensor_encoded[i][j * bank_size: (j + 1) * bank_size][
                    sparse_res['sorted_topk_indices']] = -8 * torch.ones(nnz, dtype=torch.int8)
            elif lp_dtype == INT8:
                lp_tensor_encoded[i][j * bank_size: (j + 1) * bank_size][
                    sparse_res['sorted_topk_indices']] = -128 * torch.ones(nnz, dtype=torch.int8)

            hp_tensor[i][j * nnz: (j + 1) * nnz] = hp_sparse_qnt['qnt_block']

            hp_scale[i][j] = scale_hp_fp24
            lp_scale[i][j] = scale_lp_fp24
            bitmasks[i][j] = bm

            # for test
            hp_tensor_zeros = hp_tensor_zeros.to(torch.float32)  # fp8e43 do not support fp8e43

            lp_tensor_zeros[i][j * bank_size: (j + 1) * bank_size][sparse_res['lp_indices']] = lp_sparse_qnt[
                'qnt_block']
            hp_tensor_zeros[i][j * bank_size: (j + 1) * bank_size][sparse_res['sorted_topk_indices']] = hp_sparse_qnt[
                'qnt_block'].to(torch.float32)
    # print(hp_tensor[i])
    # print(hp_scale[i])
    # print(hp_tensor_zeros[i])
    # print(hp_tensor_zeros[i] * hp_scale[i])
    # print(input_matrix[i])
    # print("1111111111111111111111111111111")
    # print(lp_tensor_encoded[i])
    # print(lp_scale[i])
    # print(lp_tensor_zeros[i])
    # print(lp_tensor_zeros[i] * lp_scale[i])
    # print(input_matrix[i])

    for i in range(k):
        for j in range(bank_num):
            bank = weight_tensor[i][j * bank_size: (j + 1) * bank_size].clone()
            bank = bank.to(torch.float32)
            weight_qnt_res = bank_quantize(bank, dtype_torch_map.get(BF16), weight_dtype)
            weight_qnt[i][j * bank_size: (j + 1) * bank_size] = weight_qnt_res['qnt_block']
            weight_scale[i][j] = weight_qnt_res['scale']

    # for test
    # print(weight_qnt[i])
    # print(weight_scale[i])
    # print(weight_tensor[i])
    # print(weight_qnt[i] * weight_scale[i])

    hp_dqnt = hp_tensor_zeros.to(torch.float32).reshape(w, bank_num, bank_size) * hp_scale.unsqueeze(-1)
    lp_dqnt = lp_tensor_zeros.to(torch.float32).reshape(w, bank_num, bank_size) * lp_scale.unsqueeze(-1)
    weight_dqnt = weight_qnt.to(torch.float32).reshape(k, bank_num, bank_size) * weight_scale.unsqueeze(-1)
    res_host = hp_dqnt.reshape(w, c) @ weight_dqnt.reshape(k, c).T + lp_dqnt.reshape(w, c) @ weight_dqnt.reshape(k, c).T
    res_host = res_host.to(dtype_torch_map.get(out_dtype))

    # for cmodel seg=2
    if nnz == 8:
        if c != 128:
            raise ValueError(f"c must equal to 128 when nnz=8(for test)")
        else:
            weight_qnt = dma_format_convert(k, c, weight_qnt, 64, 1)
            weight_scale = dma_format_convert(k, 2, weight_scale, 1, 1)
            hp_scale = dma_format_convert(w, 2, hp_scale, 1, 1)
            lp_tensor_encoded = dma_format_convert(w, c, lp_tensor_encoded, 64, 1)
            lp_scale = dma_format_convert(w, 2, lp_scale, 1, 1)

    return {
        'input_tensor': input_matrix,
        'hp_tensor': hp_tensor,
        'lp_tensor_encoded': lp_tensor_encoded,
        'weight_qnt_tensor': weight_qnt,
        'hp_scale': hp_scale,
        'lp_scale': lp_scale,
        'weight_scale': weight_scale,
        'res_host': res_host,
        'weight_tensor': weight_tensor,
        'hp_index': sort_index
    }


def gen_data_dense_with_scale(
        w: int,
        c: int,
        k: int,
        input_dtype: str,
        weight_dtype: str,
) -> Dict[str, Any]:
    if input_dtype not in (INT8, FP8E5M2, FP8E4M3) or weight_dtype not in (INT8, FP8E5M2, FP8E4M3):
        raise ValueError(f"only support int8, fp8 for lp gemm")
    bank_size = 64
    bank_num = int(c / bank_size)
    input_tensor = generate_matrix(w, c, torch.bfloat16)
    weight_tensor = generate_matrix(k, c, torch.bfloat16)

    input_qnt = torch.zeros(w, c, dtype=dtype_torch_map[input_dtype])
    input_scale = torch.zeros(w, bank_num, dtype=torch.float32)
    weight_qnt = torch.zeros(k, c, dtype=dtype_torch_map[weight_dtype])
    weight_scale = torch.zeros(k, bank_num, dtype=torch.float32)

    for i in range(w):
        for j in range(bank_num):
            block = input_tensor[i][j * bank_size: (j + 1) * bank_size].clone()
            block = block.to(torch.float32)
            block_qnt = bank_quantize(block, torch.bfloat16, input_dtype)
            input_qnt[i][j * bank_size: (j + 1) * bank_size] = block_qnt['qnt_block']
            input_scale[i][j] = torch.from_numpy(float32_to_bf24_as_float32(block_qnt['scale'].numpy()))

    for i in range(k):
        for j in range(bank_num):
            block = weight_tensor[i][j * bank_size: (j + 1) * bank_size].clone()
            block = block.to(torch.float32)
            block_qnt = bank_quantize(block, torch.bfloat16, weight_dtype)
            weight_qnt[i][j * bank_size: (j + 1) * bank_size] = block_qnt['qnt_block']
            weight_scale[i][j] = torch.from_numpy(float32_to_bf24_as_float32(block_qnt['scale'].numpy()))

    input_dqnt = input_qnt.to(torch.float32).reshape(w, bank_num, bank_size) * input_scale.unsqueeze(-1)
    weight_dqnt = weight_qnt.to(torch.float32).reshape(k, bank_num, bank_size) * weight_scale.unsqueeze(-1)
    res_host = input_dqnt.reshape(w, c) @ weight_dqnt.reshape(k, c).T
    res_host = res_host.to(torch.bfloat16)
    return {
        'input_tensor': input_qnt,
        'weight_tensor': weight_qnt,
        'input_scale': input_scale,
        'weight_scale': weight_scale,
        'res_host': res_host,
    }


def spu_host_data(
        matmul_mode: str,
        left_matrix_w_c_w: str,
        matrix_c: str,
        right_matrix_k_c_k: str,
        input_file_0: str,
        input_file_1: str,
        output_file_0: str,
        input_file_2: str,
        nnz_c: str,
        case_data_description: str,
        in0_data_type_str: str,
        in1_data_type_str: str,
        in2_data_type_str: str,
        out0_data_type_str: str,
        input_file_3: str,
        input_file_4: str,
        input_file_5: str,
        in3_data_type_str: str,
        in4_data_type_str: str,
        in5_data_type_str: str
) -> bool:
    if matmul_mode.lower() == 'sparse_mask':
        if in0_data_type_str.lower() == INT8:
            input_dtype = torch.int8
        elif in0_data_type_str.lower() == BF16:
            input_dtype = torch.bfloat16
        elif in0_data_type_str.lower() == FP8E4M3:
            input_dtype = torch.float8_e4m3fn
        elif in0_data_type_str.lower() == FP8E5M2:
            input_dtype = torch.float8_e5m2
        else:
            raise ValueError(f"unsupported input_data type")

        if in1_data_type_str.lower() == INT8:
            weight_dtype = torch.int8
        elif in1_data_type_str.lower() == BF16:
            weight_dtype = torch.bfloat16
        elif in1_data_type_str.lower() == FP8E4M3:
            weight_dtype = torch.float8_e4m3fn
        elif in1_data_type_str.lower() == FP8E5M2:
            weight_dtype = torch.float8_e5m2
        else:
            raise ValueError(f"unsupported weight_data type")

        if in2_data_type_str.lower() not in (UINT64, UINT32):
            raise ValueError(f"unsupported bitmask type")

        if out0_data_type_str.lower() == INT8:
            output_dtype = torch.int8
        elif out0_data_type_str.lower() == BF16:
            output_dtype = torch.bfloat16
        else:
            raise ValueError(f"unsupported output_data type")

        w = int(left_matrix_w_c_w)
        c = int(matrix_c)
        k = int(right_matrix_k_c_k)
        nnz = int(nnz_c)
        bank_size = 64
        if c <= 64:
            bank_size = c
        data = gen_data_sparse_mask(w, k, c, nnz, bank_size, input_dtype, weight_dtype,
                                    output_dtype)
        output = data['output']
        input_sparse_nnz = data['input_sparse_nnz']
        weight_dense = data['weight_dense']
        bitmasks = data['bitmasks']

        save_tensor_as_decimal_txt(input_sparse_nnz.reshape(1, -1), input_file_0)
        save_tensor_as_decimal_txt(weight_dense.reshape(1, -1), input_file_1)
        save_tensor_as_decimal_txt(output.reshape(1, -1), output_file_0)
        np.savetxt(input_file_2, bitmasks.reshape(1, -1), fmt='%d')

    elif matmul_mode.lower() == 'sparse_hp_lp':
        if in0_data_type_str.lower() not in (INT8, BF16, FP8E5M2, FP8E4M3):
            raise ValueError(f"unsupported hp_tensor dtype")
        if in1_data_type_str.lower() not in (FP32, NA):
            raise ValueError(f"unsupported hp_scale dtype")
        if in2_data_type_str.lower() not in (INT8, INT4):
            raise ValueError(f"unsupported lp_tensor dtype")
        if in3_data_type_str.lower() not in (FP32, NA):
            raise ValueError(f"unsupported lp_scale dtype")
        if in4_data_type_str.lower() not in INT8:
            raise ValueError(f"unsupported weight_tensor dtype")
        if in5_data_type_str.lower() not in (FP32, NA):
            raise ValueError(f"unsupported weight_scale dtype")

        w = int(left_matrix_w_c_w)
        c = int(matrix_c)
        k = int(right_matrix_k_c_k)
        nnz = int(nnz_c)

        data = gen_data_sparse_hp_lp(w, k, c, nnz, 64, in0_data_type_str, in2_data_type_str, in4_data_type_str)

        res_host = data['res_host']
        hp_tensor_nnz = data['hp_tensor']
        lp_tensor_encoded = data['lp_tensor_encoded']
        hp_scale = data['hp_scale']
        lp_scale = data['lp_scale']
        weight_tensor = data['weight_qnt_tensor']
        weight_scale = data['weight_scale']

        save_tensor_as_decimal_txt(hp_tensor_nnz.reshape(1, -1), input_file_0)
        if in1_data_type_str.lower() != NA:  # hp is bf16 no scale
            save_tensor_as_decimal_txt(hp_scale.reshape(1, -1), input_file_1)
        save_tensor_as_decimal_txt(lp_tensor_encoded.reshape(1, -1), input_file_2)
        if in3_data_type_str.lower() != NA:  # hp is bf16 no scale
            save_tensor_as_decimal_txt(lp_scale.reshape(1, -1), input_file_3)
        save_tensor_as_decimal_txt(weight_tensor.reshape(1, -1), input_file_4)
        save_tensor_as_decimal_txt(weight_scale.reshape(1, -1), input_file_5)
        save_tensor_as_decimal_txt(res_host.reshape(1, -1), output_file_0)

    elif matmul_mode.lower() == 'dense':
        if in0_data_type_str.lower() not in (INT8, BF16, FP8E5M2, FP8E4M3):
            raise ValueError(f"unsupported input dtype")
        if in1_data_type_str.lower() not in (INT8, BF16, FP8E5M2, FP8E4M3):
            raise ValueError(f"unsupported weight dtype")

        w = int(left_matrix_w_c_w)
        c = int(matrix_c)
        k = int(right_matrix_k_c_k)
        input_tensor = generate_matrix(w, c, dtype_torch_map[in0_data_type_str])
        weight_tensor = generate_matrix(k, c, dtype_torch_map[in1_data_type_str])

        res_host = input_tensor.to(torch.float32) @ weight_tensor.to(torch.float32).T
        res_host = res_host.to(dtype_torch_map[out0_data_type_str])

        save_tensor_as_decimal_txt(input_tensor.reshape(1, -1), input_file_0)
        save_tensor_as_decimal_txt(weight_tensor.reshape(1, -1), input_file_1)
        save_tensor_as_decimal_txt(res_host.reshape(1, -1), output_file_0)

    elif matmul_mode.lower() == 'dense_with_scale':
        if in0_data_type_str.lower() not in (INT8, FP8E5M2, FP8E4M3):
            raise ValueError(f"unsupported input dtype")
        if in1_data_type_str.lower() not in (INT8, FP8E5M2, FP8E4M3):
            raise ValueError(f"unsupported weight dtype")

        if in0_data_type_str.lower() != in1_data_type_str.lower():
            raise ValueError(f"type of input must be same as weight")

        w = int(left_matrix_w_c_w)
        c = int(matrix_c)
        k = int(right_matrix_k_c_k)

        data = gen_data_dense_with_scale(w, c, k, in0_data_type_str.lower(), in1_data_type_str.lower())

        input_qnt = data['input_tensor']
        weight_qnt = data['weight_tensor']
        input_scale = data['input_scale']
        weight_scale = data['weight_scale']
        res_host = data['res_host']

        save_tensor_as_decimal_txt(input_qnt.reshape(1, -1), input_file_0)
        save_tensor_as_decimal_txt(weight_qnt.reshape(1, -1), input_file_1)
        save_tensor_as_decimal_txt(input_scale.reshape(1, -1), input_file_2)
        save_tensor_as_decimal_txt(weight_scale.reshape(1, -1), input_file_3)
        save_tensor_as_decimal_txt(res_host.reshape(1, -1), output_file_0)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPU Host Case Data Generation")
    parser.add_argument("-e", type=str, required=True, help="Engine type，例如 spu")
    parser.add_argument("-matmul_mode", type=str, help="矩阵乘模式，例如 sparse_mask / sparse_hp_lp")
    parser.add_argument("-left_matrix_w_c_w", type=int, help="左矩阵 W 大小")
    parser.add_argument("-matrix_c", type=int, help="矩阵 C 大小")
    parser.add_argument("-right_matrix_k_c_k", type=int, help="右矩阵 K 大小")
    parser.add_argument("-nnz_c", type=int, help="非零元素数量")
    parser.add_argument("-case_data_description", type=str, help="用例描述，比如 case001")
    parser.add_argument("-in0", type=str, help="input0 文件路径")
    parser.add_argument("-in0_data_type_str", type=str, help="input0 数据类型")
    parser.add_argument("-in1", type=str, help="input1 文件路径")
    parser.add_argument("-in1_data_type_str", type=str, help="input1 数据类型")
    parser.add_argument("-in2", type=str, help="input2 文件路径")
    parser.add_argument("-in2_data_type_str", type=str, help="input2 数据类型")
    parser.add_argument("-out0", type=str, help="output0 文件路径")
    parser.add_argument("-out0_data_type_str", type=str, help="output0 数据类型")
    # sparse_hp_lp 模式新增的额外输入（稀疏 + 高低精度混合的时候需要）
    parser.add_argument("-in3", type=str, default="", help="input3 文件路径 (只在 sparse_hp_lp 模式需要)")
    parser.add_argument("-in3_data_type_str", type=str, default="", help="input3 数据类型")
    parser.add_argument("-in4", type=str, default="", help="input4 文件路径 (只在 sparse_hp_lp 模式需要)")
    parser.add_argument("-in4_data_type_str", type=str, default="", help="input4 数据类型")
    parser.add_argument("-in5", type=str, default="", help="input5 文件路径 (只在 sparse_hp_lp 模式需要)")
    parser.add_argument("-in5_data_type_str", type=str, default="", help="input5 数据类型")
    parser.add_argument("-st", type=str, default="", help="dtypr for sim")
    args = parser.parse_args()

    if args.e.lower() == "spu":
        spu_host_data(
            matmul_mode=args.matmul_mode,
            left_matrix_w_c_w=args.left_matrix_w_c_w,
            matrix_c=args.matrix_c,
            right_matrix_k_c_k=args.right_matrix_k_c_k,
            input_file_0=args.in0,
            input_file_1=args.in1,
            output_file_0=args.out0,
            input_file_2=args.in2,
            nnz_c=args.nnz_c,
            case_data_description=args.case_data_description,
            in0_data_type_str=args.in0_data_type_str,
            in1_data_type_str=args.in1_data_type_str,
            in2_data_type_str=args.in2_data_type_str,
            out0_data_type_str=args.out0_data_type_str,
            input_file_3=args.in3,
            input_file_4=args.in4,
            input_file_5=args.in5,
            in3_data_type_str=args.in3_data_type_str,
            in4_data_type_str=args.in4_data_type_str,
            in5_data_type_str=args.in5_data_type_str,
        )
    elif args.e.lower() == 'sim':
        sim_bin(bin_file_path1=args.in0,
                bin_file_path2=args.in1,
                dtype=args.st
                )
    pass
