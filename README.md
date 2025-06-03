
# data generatation tool

> 计算单元自动生成 **Host 侧标定数据** 的 Python3 脚本  
> 支持 **Dense / Sparse-Mask / Sparse HP-LP / Dense-With-Scale** 多种模式，覆盖 **INT4 / INT8 / FP8 / BF16 / FP32** 等常用张量格式，方便硬件仿真验证。


## Envirnment

| 依赖 | 版本 |
|---|---|
| Python | 3.8 – 3.12 |
| PyTorch | ≥ 2.1（需包含 `torch.float8_e4m3fn / torch.float8_e5m2` 支持） |
| NumPy | ≥ 1.23 |


## install

```bash
git clone https://github.com/ouou-haha/spu_host_cal.git
cd spu_host_cal
git checkout package
pip3 install -e .
```


