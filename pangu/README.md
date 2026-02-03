# openPangu-Embedded on Ascend NPU (Inference / LoRA / Merge)

This folder provides a **practical workflow** for running **openPangu-Embedded** on **Ascend NPU**, including:

- Inference sanity-check (print NPU info + model architecture + a short generation)
- LoRA fine-tuning (torchrun + HCCL DDP)
- Merging LoRA adapters back into the base model (export a merged HF-style directory)
- Reproducing our evaluation pipeline (generate JSONL predictions + score against gold)

> This repository does **NOT** distribute openPangu model weights. Please obtain the model from the official channel and follow the model license.

---

## Layout

```text
pangu/
├── inference/
│   ├── run_pangu.py                # sanity-check inference (base model)
│   ├── run_pangu.sh                # wrapper (tee logs)
│   └── run_merged_pangu.py         # inference on a merged model -> JSONL predictions
├── scripts/
│   ├── train_lora_pangu_ddp.py      # LoRA fine-tuning (NPU, torchrun)
│   └── merge_lora_to_model.py       # merge LoRA -> base model
└── eval/
    └── eval_metrics.py              # scoring script (BLEU/GLEU/METEOR/ROUGE/BERTScore*)
```

---

## 1) Environment

Example (adjust to your system):

```bash
conda activate pangu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

Recommended NPU allocator settings:

```bash
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=8
```

---

## 2) Base-model inference sanity-check

```bash
cd pangu/inference
bash run_pangu.sh /opt/pangu/openPangu-Embedded-1B-V1.1
```

The log will be saved as `run_pangu_full.log`.

---

## 3) LoRA fine-tuning (torchrun + HCCL DDP)

### Data format

`train_lora_pangu_ddp.py` expects JSON/JSONL samples with:

- `instruction`
- `input` (optional)
- `output`

### Example (8 NPUs)

```bash
torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29500 \
  pangu/scripts/train_lora_pangu_ddp.py \
  --model_dir /opt/pangu/openPangu-Embedded-1B-V1.1 \
  --train_jsonl /opt/pangu_demo/train.jsonl \
  --out_dir /opt/pangu_demo/pangu_lora_adapter \
  --epochs 10 \
  --micro_bsz 8 \
  --grad_accum 8 \
  --max_len 2048 \
  --lr 5e-6
```

---

## 4) Merge LoRA into the base model

```bash
python pangu/scripts/merge_lora_to_model.py \
  --model_dir /opt/pangu/openPangu-Embedded-1B-V1.1 \
  --adapter_dir /opt/pangu_demo/pangu_lora_adapter \
  --out_dir /opt/pangu/openPangu-Embedded-1B-V1.1-lora-merged \
  --dtype bfloat16 \
  --device cpu
```

The output directory is a standard HuggingFace model folder you can load directly.

---

## 5) Reproduce evaluation (merged model -> predictions -> scores)

### 5.1 Generate predictions (JSONL)

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python pangu/inference/run_merged_pangu.py \
  --model_dir /opt/pangu/openPangu-Embedded-1B-V1.1-lora-merged \
  --data_json /opt/eval_data.json \
  --out_jsonl /opt/pred.jsonl \
  --device npu \
  --dtype bfloat16 \
  --max_new_tokens 2048 \
  --temperature 0.0 \
  --local_files_only
```

Output format:

```json
{"id": 0, "answer": "..."}
```

### 5.2 Score predictions

```bash
python pangu/eval/eval_metrics.py \
  --gold_json /opt/eval_data.json \
  --pred_jsonl /opt/pred.jsonl \
  --pred_field answer \
  --match_mode order \
  --save /opt/metrics.json
```

`BERTScore` is optional (the script will skip it if the package is not installed).

---

## Chinese version

See `pangu/README.zh-CN.md`.
