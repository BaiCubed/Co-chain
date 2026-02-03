#!/usr/bin/env python


import argparse
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", type=str, default="cpu", help="cpu/cuda/npu. Default cpu is safest.")
    args = ap.parse_args()

    if str(args.device).lower() == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        try:
            import torch

            if hasattr(torch, "cuda"):
                torch.cuda.is_available = lambda: False
                torch.cuda.device_count = lambda: 0
        except Exception:
            pass

        try:
            import peft.utils.save_and_load as sal
            import safetensors.torch as st

            def _cpu_safe_load_file(filename, device=None):
                return st.load_file(filename, device="cpu")

            sal.safe_load_file = _cpu_safe_load_file
        except Exception as e:
            print(f"[WARN] Failed to patch PEFT safetensors loader: {e}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()

    if args.device and args.device != "cpu":
        model = model.to(args.device)

    model.save_pretrained(args.out_dir, safe_serialization=True)
    tok.save_pretrained(args.out_dir)

    print("[OK] Merged model saved to:", args.out_dir)


if __name__ == "__main__":
    main()
