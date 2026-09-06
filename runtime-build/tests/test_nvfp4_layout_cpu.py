"""CPU tensor tests for the real split/dequant helpers; no native kernel import."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

import torch


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise ValueError(f"missing source functions in {path}")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU tests cannot initialize CUDA")
        ns = {"torch": torch, "kE2M1ToFloat": torch.tensor(
            [0., .5, 1., 1.5, 2., 3., 4., 6.])}
        load_functions(SOURCE / "vllm/utils/torch_utils.py",
                       ["nvfp4_split_data_scale"], ns)
        load_functions(SOURCE / "tests/kernels/quantization/nvfp4_utils.py",
                       ["break_fp4_bytes", "dequant_nvfp4_kv_cache"], ns)
        cls.split = staticmethod(ns["nvfp4_split_data_scale"])
        cls.decode = staticmethod(ns["dequant_nvfp4_kv_cache"])
        cls.break_bytes = staticmethod(ns["break_fp4_bytes"])

    def fixture(self):
        pages, heads, tokens, dim = 3, 2, 8, 256
        full = dim // 2 + dim // 16
        backing = torch.full((pages, 2, heads, tokens, full), 0x7f,
                             dtype=torch.uint8)
        # Distinct SFs on every head/token/block distinguish scale ordering.
        sf_indices = torch.arange(pages * heads * tokens * 16).reshape(
            pages, heads, tokens, 16)
        scales = torch.tensor([.125, .25, .5, 1., 2., 4., 8.])[
            sf_indices.remainder(7)]
        codes = torch.arange(pages * heads * tokens * dim).reshape(
            pages, heads, tokens, dim).remainder(16).to(torch.uint8)
        packed = codes[..., ::2] | (codes[..., 1::2] << 4)
        values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                               -0., -.5, -1., -1.5, -2., -3., -4., -6.])[codes.long()]
        expected = (values.reshape(pages, heads, tokens, 16, 16)
                    * scales.unsqueeze(-1)).reshape(pages, heads, tokens, dim)
        return backing, packed, scales, expected

    def test_real_hnd_views_share_backing_and_decode_both_sides(self):
        backing, packed, scales, expected = self.fixture()
        for side, global_scale in [(0, .5), (1, 2.)]:
            data, sf = self.split(backing[:, side])
            self.assertEqual(data.untyped_storage().data_ptr(),
                             backing.untyped_storage().data_ptr())
            self.assertEqual(sf.untyped_storage().data_ptr(),
                             backing.untyped_storage().data_ptr())
            self.assertEqual(data.stride(), (4608, 1024, 128, 1))
            self.assertEqual(sf.stride(), (4608, 128, 16, 1))
            self.assertEqual(data.storage_offset(), side * 2304)
            self.assertEqual(sf.storage_offset(), side * 2304 + 2048)
            data.copy_(packed)
            sf.view(torch.uint8).copy_(scales.to(torch.float8_e4m3fn).view(torch.uint8))
            actual = self.decode(data, sf, global_scale, 256, 8, swizzle_sf=False)
            self.assertTrue(torch.equal(actual, expected * global_scale))

    def test_sparse_write_does_not_modify_other_pages_or_k_side(self):
        backing, packed, scales, expected = self.fixture()
        data, sf = self.split(backing[:, 1])
        allowed = torch.zeros_like(backing, dtype=torch.bool)
        flat_allowed = allowed.reshape(-1)
        for page, head, token in [(0, 1, 2), (2, 0, 6)]:
            data[page, head, token] = packed[page, head, token]
            sf.view(torch.uint8)[page, head, token] = scales[
                page, head, token].to(torch.float8_e4m3fn).view(torch.uint8)
            for view in (data, sf):
                start = (view.storage_offset() + page * view.stride(0)
                         + head * view.stride(1) + token * view.stride(2))
                flat_allowed[start:start + view.shape[-1]] = True
        self.assertTrue(bool((backing[~allowed] == 0x7f).all()))
        actual = self.decode(data, sf, 1., 256, 8, swizzle_sf=False)
        for page, head, token in [(0, 1, 2), (2, 0, 6)]:
            self.assertTrue(torch.equal(actual[page, head, token],
                                        expected[page, head, token]))

    def test_linear_v_and_sm100_swizzled_v_are_distinct_and_correct(self):
        backing, packed, scales, expected = self.fixture()
        data, sf = self.split(backing[:, 1])
        data.copy_(packed)
        encoded = scales.to(torch.float8_e4m3fn).view(torch.uint8)
        sf.view(torch.uint8).copy_(encoded)
        wrong = self.decode(data, sf, 1., 256, 8, swizzle_sf=True)
        self.assertFalse(torch.equal(wrong, expected))
        # Independent scalar address formula from the native SM100 writer.
        for token in range(8):
            for group in range(16):
                swizzled_t = (token // 4) * 4 + group // 4
                swizzled_s = (group % 4) * 4 + token % 4
                sf.view(torch.uint8)[:, :, swizzled_t, swizzled_s] = encoded[:, :, token, group]
        actual = self.decode(data, sf, 1., 256, 8, swizzle_sf=True)
        self.assertTrue(torch.equal(actual, expected))

    def test_all_sixteen_nibbles_and_signed_zero(self):
        packed = torch.tensor([[i | ((i + 1) << 4) for i in range(0, 16, 2)]],
                              dtype=torch.uint8)
        actual = self.break_bytes(packed, torch.float32)
        expected = torch.tensor([[0., .5, 1., 1.5, 2., 3., 4., 6.,
                                  -0., -.5, -1., -1.5, -2., -3., -4., -6.]])
        self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "checkouts/vllm")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    SOURCE = args.vllm_source.resolve()
    torch.set_num_threads(2)
    paths = [SOURCE / "vllm/utils/torch_utils.py",
             SOURCE / "tests/kernels/quantization/nvfp4_utils.py"]
    pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    began = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(LayoutTests))
    record = dict(status="PASS" if result.wasSuccessful() else "FAIL",
                  tests_run=result.testsRun, elapsed_s=time.monotonic() - began,
                  errors=[dict(test=str(t), traceback=tb) for t, tb in result.errors],
                  failures=[dict(test=str(t), traceback=tb) for t, tb in result.failures],
                  source_sha256=pins, cuda_initialized=torch.cuda.is_initialized(),
                  gpu_kernel_tested=False, model_quality_tested=False)
    if record["cuda_initialized"]:
        record["status"] = "FAIL"
    args.json_out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    sys.exit(record["status"] != "PASS")
