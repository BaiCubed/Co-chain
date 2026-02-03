#!/usr/bin/env python
import argparse
import os
import sys
import time

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None  # type: ignore

from transformers import AutoTokenizer, AutoModelForCausalLM


def _print_env():
    print("python:", sys.version.replace("\n", " "))
    print("torch:", getattr(torch, "__version__", "unknown"))
    if torch_npu is not None:
        print("torch_npu: available")
    else:
        print("torch_npu: not imported (OK if running on CPU/CUDA)")
    if hasattr(torch, "npu"):
        try:
            cnt = torch.npu.device_count()
            print("npu_device_count:", cnt)
            for i in range(cnt):
                try:
                    name = torch.npu.get_device_name(i)
                except Exception:
                    name = "unknown"
                print(f"npu[{i}]:", name)
        except Exception as e:
            print("npu_info_error:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="Path to openPangu-Embedded model directory.")
    ap.add_argument("--device", type=str, default="npu:0", help="npu:0 / cpu / cuda:0")
    ap.add_argument("--prompt", type=str, default="请用一句话解释什么是产业技术风险。")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--log_file", type=str, default="", help="If set, tee prints to this file.")
    args = ap.parse_args()

    _print_env()

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if "npu" in args.device else None,
    )

    model = model.to(args.device)
    model.eval()

    print("\n===== model structure =====")
    print(model)

    inputs = tok(args.prompt, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )

    text = tok.decode(out[0], skip_special_tokens=True)
    print("\n===== prompt =====")
    print(args.prompt)
    print("\n===== output =====")
    print(text)

    print("\n[OK] elapsed_sec:", round(time.time() - t0, 2))


if __name__ == "__main__":
    main()
