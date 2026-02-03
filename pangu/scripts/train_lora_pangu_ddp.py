import os, json, argparse
import time
import torch
import torch_npu
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


def init_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    return local_rank, rank, world_size


def is_rank0(rank):
    return rank == 0


def guess_target_modules(model):
    common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "wq", "wk", "wv", "wo"]
    present = set()
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            present.add(n.split(".")[-1])

    picked = [x for x in common if x in present]
    if picked:
        return picked

    picked = set()
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            ln = n.lower()
            if ("attn" in ln) or ("attention" in ln) or ("mlp" in ln) or ("ffn" in ln):
                picked.add(n.split(".")[-1])
    return sorted(picked)


class JsonlSFT_B(Dataset):
    """支持：
    - JSONL: 一行一个 dict
    - JSON: 顶层为 list，或 dict 里包含 train/data/任意 list 字段
    格式B字段：instruction/input/output
    """
    def __init__(self, path, tokenizer, max_len=2048, json_key=None):
        self.tok = tokenizer
        self.max_len = max_len
        self.rows = []

        with open(path, "r", encoding="utf-8") as f:
            head = f.read(2048)
        stripped = head.lstrip()

        if stripped.startswith("{") or stripped.startswith("["):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            if isinstance(obj, list):
                rows = obj
            elif isinstance(obj, dict):
                if json_key and json_key in obj and isinstance(obj[json_key], list):
                    rows = obj[json_key]
                elif "train" in obj and isinstance(obj["train"], list):
                    rows = obj["train"]
                elif "data" in obj and isinstance(obj["data"], list):
                    rows = obj["data"]
                else:
                    rows = None
                    for v in obj.values():
                        if isinstance(v, list):
                            rows = v
                            break
                    if rows is None:
                        raise ValueError("JSON 顶层是 dict，但找不到任何 list 字段（train/data/其它list都没有）")
            else:
                raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")

        else:
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    rows.append(json.loads(s))

        for r in rows:
            if not isinstance(r, dict):
                continue
            r.setdefault("instruction", "")
            r.setdefault("input", "")
            r.setdefault("output", "")
            self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        ins = ex.get("instruction", "")
        inp = ex.get("input", "")
        out = ex.get("output", "")

        user = ins + (("\n" + inp) if inp else "")
        prompt_text = user
        full_text = user + "\n" + out

        prompt_ids = self.tok(prompt_text, truncation=True, max_length=self.max_len)["input_ids"]
        full = self.tok(full_text, truncation=True, max_length=self.max_len)
        input_ids = full["input_ids"]
        attn = full["attention_mask"]

        labels = input_ids.copy()
        pl = min(len(prompt_ids), len(labels))
        for i in range(pl):
            labels[i] = -100
        return input_ids, attn, labels


def collate(batch, pad_id):
    maxlen = max(len(x[0]) for x in batch)
    input_ids, attn, labels = [], [], []
    for ids, am, lab in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append(am + [0] * pad)
        labels.append(lab + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./pangu_lora_adapter_e10")
    ap.add_argument("--max_len", type=int, default=2048)

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro_bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_every", type=int, default=10)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "max_split_size_mb:256")

    local_rank, rank, world_size = init_dist()
    device = "npu"

    if is_rank0(rank):
        print(f"[DDP] world_size={world_size}, micro_bsz={args.micro_bsz}, grad_accum={args.grad_accum}")
        print(f"[DDP] global_batch ~= {args.micro_bsz * world_size * args.grad_accum}")

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    target_modules = guess_target_modules(model)
    if is_rank0(rank):
        print("LoRA target_modules =", target_modules)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    ds = JsonlSFT_B(args.train_jsonl, tok, max_len=args.max_len)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    dl = DataLoader(
        ds,
        batch_size=args.micro_bsz,
        sampler=sampler,
        num_workers=2,
        pin_memory=False,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    step = 0
    opt.zero_grad(set_to_none=True)

    total_batches = args.epochs * len(dl)
    if is_rank0(rank):
        t0 = time.time()
        pbar = tqdm(total=total_batches, desc="LoRA-SFT", unit="batch", dynamic_ncols=True)
    else:
        pbar = None
        t0 = None

    for ep in range(args.epochs):
        sampler.set_epoch(ep)
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                loss = model(**batch).loss / args.grad_accum

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if is_rank0(rank):
                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    opt_step = (step + 1) // args.grad_accum
                    pbar.set_postfix({
                        "epoch": ep,
                        "step": step,
                        "opt": opt_step,
                        "loss": f"{loss.item() * args.grad_accum:.4f}",
                        "elapsed_min": f"{elapsed/60:.1f}",
                    })
                pbar.update(1)

            step += 1

    if is_rank0(rank):
        pbar.close()
        total_min = (time.time() - t0) / 60.0
        print(f"[rank0] Total time: {total_min:.2f} min")

        os.makedirs(args.out_dir, exist_ok=True)
        model.module.save_pretrained(args.out_dir)
        tok.save_pretrained(args.out_dir)
        print("Saved LoRA adapter to:", args.out_dir)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()


'''
conda activate ascend_vllm092
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=8

torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29500 \
  /opt/pangu_demo/train_lora_pangu_ddp.py \
  --model_dir /opt/pangu/openPangu-Embedded-1B-V1.1 \
  --train_jsonl /opt/pangu_demo/train.jsonl \
  --out_dir /opt/pangu_demo/pangu_lora_adapter \
  --epochs 10 \
  --micro_bsz 8 \
  --grad_accum 8 \
  --max_len 2048 \
  --lr 5e-6
'''
