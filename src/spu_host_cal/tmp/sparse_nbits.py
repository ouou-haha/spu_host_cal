import torch
import torch.nn as nn
import math
from collections import namedtuple, OrderedDict
from typing import Dict, List, Tuple, Callable
from einops.einops import rearrange

import os
import torch
import matplotlib.pyplot as plt
import math


def check(x: torch.Tensor):
    if x.isinf().any():
        raise ValueError(f"x is inf| max: {x.max().item()} min: {x.min().item()}")

    if x.isnan().any():
        raise ValueError(f"x is nan| max: {x.max().item()} min: {x.min().item()}")

class MovingAvg:
    def __init__(self, rate: float) -> None:
        self.rate = rate
        self.data = None

    def update(self, data):
        if self.data is None:
            self.data = data
        else:
            self.data = self.rate * self.data + (1 - self.rate) * data
        return self

    def __repr__(self) -> str:
        return f"{self.data}"

class GlobalAvg:
    def __init__(self) -> None:
        self.count = 0
        # self.sum = 0
        self.data = 0

    def update(self, x):
        self.count += 1
        # self.sum += x
        self.data = self.data / self.count * (self.count - 1) + \
            x / self.count

    # @property
    # def data(self):
    #     return self.sum / max(self.count, 1)

class Info:
    def __init__(self, name: str, rate: float) -> None:
        self.name = name
        self.max = MovingAvg(rate=rate)
        self.min = MovingAvg(rate=rate)
        # self.mean_win = MovingAvg(rate=rate)
        # self.square_win = MovingAvg(rate=rate)
        self.mean_win = GlobalAvg()
        self.square_win = GlobalAvg()
        # self.std_win = GlobalAvg()

    def update(
        self,
        max_value: torch.Tensor,
        min_value: torch.Tensor,
        batch: torch.Tensor
    ):
        self.max.update(max_value)
        self.min.update(min_value)
        self.mean_win.update(batch.mean(list(range(batch.dim()-1))))
        self.square_win.update((batch**2).mean(list(range(batch.dim()-1))))
        # mean, std = self.get_nonzero_mean_std(batch)
        # self.mean_win.update(mean)
        # self.std_win.update(std)

    @property
    def is_key(self):
        return ".key." in self.name
        # return self.name.endswith(".key")

    @property
    def is_value(self):
        return ".value." in self.name
        # return self.name.endswith(".value")

    def get_nonzero_mean_std(self, x):
        smallest_normal = torch.finfo(x.dtype).smallest_normal
        numel = (x != 0).sum(list(range(x.dim()-1)))
        numel[torch.lt(numel,smallest_normal)] = 1

        mean = x.sum(list(range(x.dim()-1))) / numel
        var = ((x - mean) ** 2).sum(list(range(x.dim()-1))) / numel
        std  = (var + 1e-5) ** 0.5
        std[torch.lt(std,smallest_normal)] = 1
        return mean, std

    @property
    def mean(self):
        return self.mean_win.data

    @property
    def std(self):
        # return self.std_win.data
        smallest_normal = torch.finfo(self.mean.dtype).smallest_normal
        std = (self.square_win.data - self.mean ** 2).clamp(0).sqrt()
        std[std < smallest_normal] = 1
        return std

    @property
    def scales(self):
        # alpha = 0.5
        alpha = 0.95
        scale = ((self.max.data - self.min.data) / 2) ** alpha
        smallest_normal = torch.finfo(scale.dtype).smallest_normal
        scale[scale<smallest_normal] = 1
        # return scale
        return torch.ones_like(scale)
        # return scale if self.is_key else torch.ones_like(scale)

    @property
    def zp(self):
        zp = (self.max.data + self.min.data) / 2
        # return zp
        return torch.zeros_like(zp)
        # return zp if self.is_key else torch.zeros_like(zp)

    @property
    def num_bits(self):
        # num_bits={"high": 4, "low": -1}
        # num_bits={"high": 4, "low": 2}
        num_bits={"high": 4, "low": 0}
        # num_bits={"high": 4, "low": 3}
        # num_bits={"high": 8, "low": 2}
        # num_bits={"high": -1, "low": 2}
        # num_bits={"high": -1, "low": 0}
        return num_bits

    @property
    def magic_num(self):
        magic_num={"high": 1.0, "low": 1.0}
        # magic_num={"high": 1.0, "low": 0.67}
        return magic_num

    @property
    def mode(self):
        if self.is_key:
            return "per_group"
            # return "per_bank"
        elif self.is_value:
            # return "per_token"
            return "per_bank"
        else:
            return "per_bank"

    @property
    def quant_symmetric(self):
        # return False
        return False if self.is_key else True

    @property
    def quant_masked(self):
        return True

# Deprecated
'''
def calibrate(x: torch.Tensor, sparsity: float=0.5, bank_size: int=-1, dim:int=-1, mode: str="per_bank"):
    h_x = MFSparseNbits(
        sparsity=sparsity, bank_size=bank_size, dim=dim,
        num_bits={"high": -1, "low": 0}, mode=mode,
        outer_transform=False
    )(x)
    l_x = x - h_x
    # TODO: use all x for calibrate
    # l_x = x
    mask = (l_x != 0).to(l_x.dtype)

    # transpose
    dim = dim if dim >= 0 else x.dim() + dim
    if dim != x.dim() - 1:
        l_x = l_x.transpose(dim, -1)
        mask = mask.transpose(dim, -1)
    mask_shape = mask.shape

    # rearrange
    l_x = rearrange(l_x, "... c -> (...) c")
    mask = rearrange(mask, "... c -> (...) c")

    # calibrate max min
    l_x_max = l_x.max(dim=0, keepdim=True).values
    l_x_min = l_x.min(dim=0, keepdim=True).values

    # TODO: masked quantization
    ori_max, ori_min = l_x_max, l_x_min
    l_x_max = (l_x * mask + ori_min * (1 - mask)).max(dim=0, keepdim=True).values
    l_x_min = (l_x * mask + ori_max * (1 - mask)).min(dim=0, keepdim=True).values

    # shape as [1, 1, ..., c]
    l_x_max = l_x_max.view([1] * (x.dim() - 1) + [-1])
    l_x_min = l_x_min.view([1] * (x.dim() - 1) + [-1])
    # recover mask shape
    mask = mask.view(mask_shape)

    # revert transpose
    if dim != x.dim() - 1:
        l_x_max = l_x_max.transpose(dim, -1)
        l_x_min = l_x_min.transpose(dim, -1)
        mask = mask.transpose(dim, -1)

    return l_x_max, l_x_min, mask
'''

def get_calibrate_dim(layer):
    infos = {
        nn.Linear: -1,
        nn.Conv1d: -2,
        nn.Conv2d: -3,
        nn.Identity: -1,
    }
    for layer_type, dim in infos.items():
        if isinstance(layer, layer_type):
            return dim

    raise NotImplementedError(f"Unsupport layer type: {type(layer)}\n\t{infos}")



class Transform:
    supported_modes = ["per_token", "per_group", "per_bank", "per_block"]
    def __init__(self, mode: str, bank_size: int) -> None:
        self.mode = mode
        assert self.mode in self.supported_modes, f"mode {self.mode} not supported, supported modes: {self.supported_modes}"

        self.bank_size = bank_size
        if self.mode == "per_block":
            assert bank_size > 0, "bank_size must be greater than 0"

    @property
    def preprocess_funcs(self) -> Dict[str, Callable]:
        # make sure to subdivide/transpose the channel to dimension -1
        funcs = {
            "per_token": (),
            "per_group": (
                lambda x: rearrange(
                    x, "... (n g) c -> ... c n g", g=self.bank_size
                ),
            ) if self.bank_size > 0 else (
                lambda x: rearrange(x, "... s c -> ... c 1 s"),
            ),
            "per_bank": (
                lambda x: rearrange(
                    x, "... s (n b) -> ... s n b", b=self.bank_size
                ),
            ) if self.bank_size > 0 else (
                lambda x: rearrange(x, "... s c -> ... s 1 c"),
            ),
            "per_block": (
                lambda x: rearrange(
                    x,
                    "... (ns sb) (nc cb) -> ... ns nc (sb cb)",
                    sb=self.bank_size,
                    cb=self.bank_size
                ),
            ),
        }
        return funcs

    @property
    def postprocess_funcs(self) -> Dict[str, Callable]:
        # revert the preprocess_funcs to get the original shape
        funcs = {
            "per_token": (),
            "per_group": (
                lambda x: rearrange(x, "... c n g -> ... (n g) c"),
            ),
            "per_bank": (
                lambda x: rearrange(x, "... s n b -> ... s (n b)"),
            ),
            "per_block": (
                lambda x: rearrange(
                    x,
                    "... ns nc (sb cb) -> ... (ns sb) (nc cb)",
                    sb=int(math.sqrt(x.size(-1))),
                    cb=int(math.sqrt(x.size(-1))),
                ),
            )
        }
        return funcs

    def call(self, x: torch.Tensor, funcs: Tuple[Callable]) -> torch.Tensor:
        for func in funcs:
            x = func(x)
        return x

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return self.call(x, self.preprocess_funcs[self.mode])

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        return self.call(x, self.postprocess_funcs[self.mode])

class QuantTool:
    def __init__(
        self, mode: str, num_bits: int=8, bank_size: int=64,
        magic_num: float=1.0, symmetric: bool=True
    ) -> None:
        self.mode = mode
        self.num_bits = num_bits
        self.bank_size = bank_size
        self.symmetric = symmetric
        self.magic_num = magic_num
        assert num_bits >= -1, f"num_bits must be greater than or equal to -1, got {num_bits}"

        # TODO: use transform only when quant
        self.transform = Transform(mode, bank_size)
        # self.transform = Transform("per_bank", bank_size) # if transform when sparse

    def sym_quant(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # symmetric quantization
        max_ranges = x.abs().max(dim=-1, keepdim=True).values * self.magic_num
        max_int = 2 ** (self.num_bits - 1) - 1
        scales = max_ranges / max_int if self.num_bits > 1 else max_ranges
        # scales = max_ranges / (max_int + 1) if self.num_bits > 1 else max_ranges

        # set num less than smallest number to smallest number to avoid overflow
        smallest_normal = torch.finfo(x.dtype).smallest_normal * max_ranges.clamp(1)
        scales = torch.maximum(scales, smallest_normal)
        # scales[scales < smallest_normal] = smallest_normal
        # scales[scales == 0] = 1
        # quant
        x = torch.clamp(
            torch.round(x / scales),
            -max_int if self.num_bits > 1 else -1,
            max_int
        )
        return x, scales

    def sym_dequant(self, x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        x = x * scales
        return x

    def get_mean(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        if mask is None:
            mask = torch.tensor(1.0, device=x.device, dtype=x.dtype)

        if self.symmetric:
            # for symmetric quantization, mean = 0
            x_mean = torch.zeros_like(x[..., :1])
        else:
            # for asymmetric quantization
            x_max = x.max(dim=-1, keepdim=True).values
            x_min = x.min(dim=-1, keepdim=True).values

            # TODO: masked quantization
            ori_max, ori_min = x_max, x_min
            x_max = (x * mask + ori_min * (1 - mask)).max(dim=-1, keepdim=True).values
            x_min = (x * mask + ori_max * (1 - mask)).min(dim=-1, keepdim=True).values

            x_mean = (x_max + x_min) / 2

        return x_mean

    def __call__(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        if self.num_bits < 0:
            return x
        elif self.num_bits == 0:
            # zero bits
            return torch.zeros_like(x)

        # TODO: masked quantization
        if mask is None:
            mask = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        # backup the unmasked value
        x_unmasked = x * (1 - mask)

        # TODO: use transform only when quant
        x = self.transform.preprocess(x)
        mask = self.transform.preprocess(mask) if mask.dim() > 1 else mask

        # subtract mean
        x_mean = self.get_mean(x, mask)
        x -= x_mean
        # quant & dequant
        quanted_x, scales = self.sym_quant(x * mask)
        dequanted_x = self.sym_dequant(quanted_x, scales)

        # add back mean
        dequanted_x += x_mean

        # TODO: use transform only when quant
        dequanted_x = self.transform.postprocess(dequanted_x)
        mask = self.transform.postprocess(mask) if mask.dim() > 1 else mask

        # TODO: masked quantization: restore unmasked value
        dequanted_x = dequanted_x * mask + x_unmasked * (1 - mask)
        return dequanted_x

class MFSparse:
    def __init__(
        self,
        sparsity: float=0.5,
        bank_size: int=-1,
        mode: str="per_bank",
    ) -> None:
        self.sparsity = sparsity
        self.bank_size = bank_size
        self.transform = Transform(mode, bank_size)

    def get_topk_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0:
            return torch.ones_like(x, dtype=torch.bool)

        k = int(x.size(-1) * (1 - self.sparsity))
        topk_values, topk_indices = x.abs().topk(k=k, dim=-1)
        # self.plot(topk_indices)

        # robust form
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        # mask = x.abs() >= topk_values[..., -1:]
        return mask

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sparsity <= 0:
            return x, torch.zeros_like(x), torch.ones_like(x)

        # rearrange x to bank size
        # bank_size = self.bank_size if self.bank_size > 0 else x.shape[-1]
        # x = rearrange(x, "... (n b) -> ... n b", b=bank_size)
        x = self.transform.preprocess(x)

        # split high & low
        mask = self.get_topk_mask(x).to(x.dtype)

        # revert x shape
        # x = rearrange(x, "... n b -> ... (n b)")
        # mask = rearrange(mask, "... n b -> ... (n b)")
        x = self.transform.postprocess(x)
        mask = self.transform.postprocess(mask)

        x_high = x * mask
        x_low = x * (1 - mask)

        return x_high, x_low, mask

class MFSparseNbits:
    def __init__(
        self,
        bank_size: int=-1,
        sparsity: float=0.5,
        sparse_mode: str="per_bank",
        quant_mode: str="per_bank",
        num_bits: Dict={"high": -1, "low": -1},
        magic_num: Dict={"high": 1.0, "low": 1.0},
        quant_symmetric: bool=True,
        quant_masked: bool=True,
        zp: torch.Tensor=None,
        scales: torch.Tensor=None,
    ) -> None:
        """
        Args:
            bank_size (int, optional): given bank size.
            sparsity (float, optional): sparsity.
            sparse_mode (str, optional): Options list as follows:
                - per_bank: quantize each bank separately, which operates in the feature dimension with given bank_size.
                - per_token: quantize each token separately, which operates in the feature dimension with `bank_size = -1`.
                - per_group: quantize each group separately, which operates in the sequence dimension with `group_size = bank_size`.
                - per_block: quantize each block separately, which operates both in the feature and sequence dimension with given bank_size and `group_size = bank_size`.
            quant_mode (str, optional): set to the same as `sparse_mode` generally, hardware requires that sparse_mode and quant_mode operates in the same dimension.
            num_bits (Dict, optional): number of bits to quantize, -1 means no quantize.
            magic_num (Dict, optional): set `magic_num * max_range` as the quantized max value.
            quant_symmetric (bool, optional): quantize symmetric or asymmetric.
            quant_masked (bool, optional): quantize within mask.
            zp (torch.Tensor, optional): zero point of linear transform before sparse and quantize. Set to None to disable.
            scales (torch.Tensor, optional): scale of linear transform before sparse and quantize. Set to None to disable.
        """
        super().__init__()
        self.sparsity = sparsity
        assert 0 <= self.sparsity <= 1, f"sparsity {self.sparsity} is not in [0, 1]"
        self.bank_size = bank_size
        self.sparse_mode = sparse_mode
        self.quant_mode = quant_mode
        self.num_bits = num_bits
        self.magic_num = magic_num
        self.zp = zp if zp is not None else torch.zeros(1)
        self.scales = scales if scales is not None else torch.ones(1)
        self.quant_symmetric = quant_symmetric
        self.quant_masked = quant_masked

        self.high_quant_tool = QuantTool(
            mode=quant_mode, num_bits=num_bits["high"], bank_size=bank_size,
            magic_num=magic_num["high"], symmetric=quant_symmetric,
        )
        self.low_quant_tool = QuantTool(
            mode=quant_mode, num_bits=num_bits["low"], bank_size=bank_size,
            magic_num=magic_num["low"], symmetric=quant_symmetric,
        )

        # sparse tool
        self.sparse_tool = MFSparse(
            sparsity=sparsity, bank_size=bank_size, mode=sparse_mode
        )

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.zp.to(x.device, x.dtype)) / self.scales.to(x.device, x.dtype)

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scales.to(x.device, x.dtype) + self.zp.to(x.device, x.dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or \
                (self.sparsity <= 0 and self.high_quant_tool.num_bits < 0):
            return x

        # TODO: plot, to be deleted
        ori_x = x

        # norm
        x = self.norm(x)

        # split x_high & x_low
        x_high, x_low, mask = self.sparse_tool(x)

        # TODO: plot, to be deleted
        ori_x_high, ori_x_low = x_high, x_low

        # quant
        x_high = self.high_quant_tool(x_high, mask=mask if self.quant_masked else None)
        x_low = self.low_quant_tool(x_low, mask=(1-mask) if self.quant_masked else None)

        # merge x_high & x_low
        x = x_high * mask + x_low * (1 - mask)

        # denorm
        x = self.denorm(x)

        return x

    @property
    def is_seq_rely(self):
        return bool(
            set([self.sparse_mode, self.quant_mode]) & \
                set(["per_group", "per_block"])
        )

    def print_stats(self):
        print(
            f"| bank_size: {self.bank_size} | "
            f"sparsity: {self.sparsity} | "
            f"sparse_mode: {self.sparse_mode} | "
            f"quant_mode: {self.quant_mode} | "
            f"num_bits: {self.num_bits} | "
            f"magic_num: {self.magic_num} | "
            f"quant_symmetric: {self.quant_symmetric} | "
            f"quant_masked: {self.quant_masked} | "
            ,
            flush=True
        )

if __name__ == "__main__":
    bank_size=64
    sparsity=0
    quant_mode="per_group"
    num_bits={"high": 4, "low": 0}
    quant_symmetric=True
    quant_masked=True

    # x = torch.randn(4, 64, 128) # .uniform_(-10, 10)
    # x -= x.abs().max() + 10
    # x -= 1
    x = torch.randn(1, 64, dtype=torch.bfloat16).reshape(64, 1)
    tool = MFSparseNbits(
        sparsity=sparsity,
        bank_size=bank_size,
        num_bits=num_bits,
        # sparse_mode=sparse_mode,
        quant_mode=quant_mode,
        quant_symmetric=False,
        quant_masked=True,
    )
    #print(f"x:{x.reshape(1, 64)}")
    y = tool(x)
    #print(f"mean: {(y - x).abs().mean():.4f}\tmax: {(y - x).abs().max():.4f}")
    # print(x.abs() - y.abs())
    # print(f"mean: {x.abs().mean():.4f}\tmax: {x.abs().max():.4f}")

