import unittest
import tempfile
import os
import numpy as np
import torch

import spu_host_cal.data_tool

from spu_host_cal.data_tool import (
    load_bin_tensor, save_tensor_bin, sim, sim_bin, dma_format_convert,
    load_int4_from_bin, save_int4_as_bin, float32_to_bf24_as_float32,
    generate_matrix, save_tensor_as_decimal_txt, get_topk_index,
    bank_quantize, gen_data_d2sqnt,
    gen_data_sparse_mask, gen_data_sparse_hp_lp,
    gen_data_dense_with_scale, spu_host_data, gen_data_topk,
    gen_data_s2ddqnt, s2ddqnt, gen_data_qnt
)
from tmp.sparse_nbits import (
    MFSparseNbits
)


class TestDataTool(unittest.TestCase):
    def test_generate_matrix(self):
        m = generate_matrix(2, 3, torch.int8)
        self.assertEqual(m.shape, (2, 3))
        self.assertEqual(m.dtype, torch.int8)
        m2 = generate_matrix(4, 5, torch.float32)
        self.assertEqual(m2.dtype, torch.float32)

    def test_float32_to_bf24(self):
        arr = np.array([1.2345, -2.3456], dtype=np.float32)
        out = float32_to_bf24_as_float32(arr)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.float32)
        bits_in = arr.view(np.uint32)
        bits_out = out.view(np.uint32)
        # lower 8 bits should be zero
        self.assertTrue(np.all((bits_out & 0xFF) == 0))
        self.assertTrue(np.all((bits_out & ~0xFF) == (bits_in & ~0xFF)))

    def test_int4_load_save(self):
        tensor = torch.tensor([-8, -1, 0, 7], dtype=torch.int8)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        save_int4_as_bin(tensor, path)
        loaded = load_int4_from_bin(path)
        os.remove(path)
        self.assertTrue(torch.equal(loaded, tensor))

    def test_sim(self):
        A = torch.randn(10)
        self.assertAlmostEqual(sim(A, A), 1.0, places=5)
        self.assertAlmostEqual(sim(A, -A), -1.0, places=5)

    def test_sim_bin(self):
        data = (torch.randn(5).to(torch.bfloat16)).view(torch.float16).numpy()
        with tempfile.NamedTemporaryFile(delete=False) as f1, tempfile.NamedTemporaryFile(delete=False) as f2:
            p1, p2 = f1.name, f2.name
        data.tofile(p1)
        data.tofile(p2)
        result = sim_bin(p1, p2, "bf16")
        os.remove(p1)
        os.remove(p2)
        self.assertTrue(result)

    def test_dma_format_convert(self):
        tensor = torch.randn(16, 128).reshape(16, 128)
        out2 = torch.zeros_like(tensor).reshape(32, 64)
        out2[0:16, 0:64] = tensor[:, 0:64]
        out2[16:32, 0:64] = tensor[:, 64:128]
        out1 = dma_format_convert(16, 128, tensor, 64, 1)
        # self.assertEqual(out1.reshape(32, 64), out2)

    def test_load_bin_tensor_and_save_tensor_bin(self):
        # FP32 round-trip
        arr = np.random.randn(4).astype(np.float32)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        arr.tofile(path)
        loaded = load_bin_tensor(path, "fp32")
        os.remove(path)
        self.assertTrue(torch.allclose(loaded, torch.from_numpy(arr)))

        # INT8 save and load round-trip
        t = torch.randint(-10, 10, (5,), dtype=torch.int8)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            outp = tmp.name
        save_tensor_bin(t, outp, "int8")
        loaded_int8 = torch.from_numpy(np.fromfile(outp, dtype=np.int8))
        os.remove(outp)
        self.assertTrue(torch.equal(loaded_int8, t.flatten()))

    def test_save_tensor_as_decimal_txt(self):
        t = torch.tensor([1.5, 2.5, -3.5])
        with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tmp:
            path = tmp.name
        save_tensor_as_decimal_txt(t, path)
        with open(path, 'r') as f:
            content = f.read().strip().split()
        os.remove(path)
        vals = [float(x) for x in content]
        self.assertEqual(vals, [1.5, 2.5, -3.5])

    def test_get_topk_index(self):
        vec = torch.tensor([3.0, 1.0, -2.0, 4.0])
        sorted_idx, topk_idx, _ = get_topk_index(vec, 2)
        self.assertTrue(torch.equal(topk_idx, torch.tensor([3, 0])))
        self.assertTrue(torch.equal(sorted_idx, torch.tensor([0, 3])))


    def test_gen_data_ds2qnt(self):
        data = gen_data_d2sqnt(64, 64, 32, 'bf16', 'int8', bank_size=64)
        self.assertIn('input_tensor', data)
        self.assertIn('output_tensor', data)
        self.assertEqual(data['input_tensor'].shape, (64, 64))

    def test_gen_data_sparse_mask(self):
        data = gen_data_sparse_mask(w=64, k=64, c=64, nnz=32, bank_size=64)
        self.assertIn('input_dense', data)
        self.assertIn('output', data)

    def test_gen_data_sparse_hp_lp(self):
        data = gen_data_sparse_hp_lp(w=64, k=64, c=64, nnz=32, bank_size=64,
                                     hp_dtype='int8', lp_dtype='int4', weight_dtype='int8', out_dtype='bf16')
        self.assertIn('hp_tensor', data)
        self.assertIn('res_host', data)

    def test_gen_data_dense_with_scale(self):
        data = gen_data_dense_with_scale(64, 64, 64, 'int8', 'int8')
        self.assertIn('input_tensor', data)
        self.assertIn('res_host', data)

    def test_spu_host_data_dense(self):
        # test dense mode
        with tempfile.NamedTemporaryFile(delete=False) as f0, tempfile.NamedTemporaryFile(
                delete=False) as f1, tempfile.NamedTemporaryFile(delete=False) as f2:
            p0, p1, p2 = f0.name, f1.name, f2.name
        result = spu_host_data(
            'dense', '2', '3', '2',
            p0, p1, p2,
            '', '', '',
            'int8', 'int8', '', 'int8',
            '', '', '', '', '', ''
        )
        self.assertTrue(result)
        self.assertTrue(os.path.getsize(p2) > 0)
        for p in (p0, p1, p2): os.remove(p)

    def test_gen_topk(self):
        data = gen_data_topk(1, 10, 5, "bf16")
        self.assertIn('input_tensor', data)
        self.assertIn('output_tensor', data)

    def test_gen_s2ddqnt(self):
        data = gen_data_s2ddqnt(64, 64, 32, "int8", "bf16")
        self.assertIn('input_sparse_qnt', data)
        self.assertIn('input_index', data)

    def test_s2ddqnt(self):
        input_data = gen_data_d2sqnt(64, 64, 32, 'bf16', 'int8', bank_size=64)
        data = s2ddqnt(input_data['output_tensor'], input_data['index'], input_data['scale'], 64, 64, 32, 'int8',
                       'bf16')
        self.assertIn('input_sparse_qnt', data)
        self.assertIn('input_index', data)

    def test_qnt(self):
        data = gen_data_qnt(64, 64, 'bf16', 'int8', bank_size=64)
        self.assertIn('input_tensor', data)
        self.assertIn('output_tensor', data)
        self.assertIn('scale', data)

    def test_quant_asym(self):
        bank_size = 64
        sparsity = 0
        quant_mode = "per_group"
        num_bits = {"high": 4, "low": 0}
        quant_symmetric = False
        quant_masked = True
        x = torch.randn(1, 64, dtype=torch.bfloat16).reshape(64, 1)
        tool = MFSparseNbits(
            sparsity=sparsity,
            bank_size=bank_size,
            num_bits=num_bits,
            # sparse_mode=sparse_mode,
            quant_mode=quant_mode,
            quant_symmetric=quant_symmetric,
            quant_masked=quant_masked,
        )
        y = tool(x)

        res = bank_quantize(x.reshape(1, 64), "int4", False)
        res_y = res['dqnt_tensor']
        error = (y.reshape(1, 64) - res_y).abs().mean()  # 均方误差

        self.assertLess(error.item(), 1e-1, "y and res_y are not close enough")
    
    def test_gen_data_topk(self):
        w, c, k = 1, 1024, 128
        data = gen_data_topk(w, c, k, 0)
        
        self.assertIn('input_tensor', data)
        self.assertIn('output_tensor', data)
        self.assertIn('index', data)


if __name__ == '__main__':
    unittest.main()
