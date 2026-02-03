
import os
import json
import argparse
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_samples(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["data", "samples", "items", "examples"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    raise ValueError("Unsupported JSON format: expected a list or a dict containing a list (data/samples/items/examples).")


def pick_text(sample: Dict[str, Any]) -> str:
    for k in ["instruction", "prompt", "question", "query", "input", "text", "content"]:
        v = sample.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return json.dumps(sample, ensure_ascii=False)


def build_prompt(tokenizer, user_text: str, system_prompt: Optional[str]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        msgs = []
        if system_prompt and system_prompt.strip():
            msgs.append({"role": "system", "content": system_prompt.strip()})
        msgs.append({"role": "user", "content": user_text})
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    if system_prompt and system_prompt.strip():
        return f"[SYSTEM]\n{system_prompt.strip()}\n\n[USER]\n{user_text}\n\n[ASSISTANT]\n"
    return f"{user_text}\n\nAnswer:\n"


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to merged model dir, e.g. /opt/pangu_demo/pangu_merged_xxx")
    ap.add_argument("--data_json", required=True, help="Input JSON (list or dict containing list)")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--npu_id", type=int, default=None, help="Which NPU to use (sets ASCEND_RT_VISIBLE_DEVICES)")
    ap.add_argument("--device", default="npu", choices=["npu", "cpu"], help="Run on NPU or CPU")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--system_prompt", type=str, default="", help="Optional system prompt")
    ap.add_argument("--local_files_only", action="store_true", help="Force offline load (recommended on your server)")
    ap.add_argument("--max_input_tokens", type=int, default=4096, help="Truncate input to this many tokens (safety)")
    ap.add_argument("--limit", type=int, default=0, help="Only run first N samples; 0 means all")
    args = ap.parse_args()

    if args.npu_id is not None:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu_id)

    torch.manual_seed(args.seed)

    local_only = True if args.local_files_only else False

    print(f"[INFO] model_dir={args.model_dir}")
    print(f"[INFO] data_json={args.data_json}")
    print(f"[INFO] out_jsonl={args.out_jsonl}")
    if args.npu_id is not None:
        print(f"[INFO] ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=local_only,
        use_fast=False,  # 你环境里经常是 slow tokenizer，先求稳
    )
    print(f"[INFO] tokenizer loaded in {time.time()-t0:.2f}s")

    if args.dtype == "auto":
        torch_dtype = "auto"
    else:
        torch_dtype = getattr(torch, args.dtype)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=local_only,
        torch_dtype=torch_dtype,
        device_map=None,  # 我们自己控制 .to()
    )
    model.eval()
    print(f"[INFO] model loaded in {time.time()-t0:.2f}s")

    if args.device == "npu":
        model = model.to("npu")
        dev = "npu"
    else:
        dev = "cpu"

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    raw = load_json(args.data_json)
    samples = iter_samples(raw)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    outputs: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        sid = s.get("id", s.get("qid", s.get("uid", i)))
        user_text = pick_text(s)
        prompt = build_prompt(tok, user_text, args.system_prompt)

        enc = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        )

        input_ids = enc["input_ids"].to(dev)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=(args.temperature > 1e-6),
            temperature=max(args.temperature, 1e-6),
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )

        with torch.inference_mode():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        gen_part = out_ids[0][input_ids.shape[1]:]
        ans = tok.decode(gen_part, skip_special_tokens=True).strip()

        row = {
            "id": sid,
            "answer": ans,
        }
        outputs.append(row)

        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            print(f"[INFO] done {i+1}/{len(samples)}", flush=True)

    write_jsonl(args.out_jsonl, outputs)
    print(f"[OK] wrote: {args.out_jsonl}  (n={len(outputs)})")


if __name__ == "__main__":
    main()
