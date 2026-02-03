# openPangu-Embedded（Ascend NPU）：推理 / LoRA 微调 / 合并

本目录提供 openPangu-Embedded 在昇腾 NPU 上的可复用流程：

- 基座模型推理自检（打印 NPU 信息 + 模型结构 + 生成结果 + 日志）
- LoRA 微调（torchrun + HCCL DDP）
- LoRA 合并回基座模型（导出 merged 目录）
- 快速复现：merged 模型生成预测 JSONL + 计算评测指标

> 本仓库不包含 openPangu 模型权重，请自行下载并遵守其 License。

## 目录结构

```text
pangu/
├── inference/
│   ├── run_pangu.py
│   ├── run_pangu.sh
│   └── run_merged_pangu.py
├── scripts/
│   ├── train_lora_pangu_ddp.py
│   └── merge_lora_to_model.py
└── eval/
    └── eval_metrics.py
```

## 环境

```bash
conda activate pangu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=8
```

## 1) 基座推理自检

```bash
cd pangu/inference
bash run_pangu.sh /opt/pangu/openPangu-Embedded-1B-V1.1
```

## 2) LoRA 微调（示例 8 卡）

```bash
torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29500 \
  pangu/scripts/train_lora_pangu_ddp.py \
  --model_dir /opt/pangu/openPangu-Embedded-1B-V1.1 \
  --train_jsonl /opt/pangu_demo/train.jsonl \
  --out_dir /opt/pangu_demo/pangu_lora_adapter \
  --epochs 10 --micro_bsz 8 --grad_accum 8 --max_len 2048 --lr 5e-6
```

## 3) 合并 LoRA

```bash
python pangu/scripts/merge_lora_to_model.py \
  --model_dir /opt/pangu/openPangu-Embedded-1B-V1.1 \
  --adapter_dir /opt/pangu_demo/pangu_lora_adapter \
  --out_dir /opt/pangu/openPangu-Embedded-1B-V1.1-lora-merged \
  --dtype bfloat16 --device cpu
```

## 4) 快速复现（生成预测 + 评测）

### 4.1 生成预测 JSONL

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python pangu/inference/run_merged_pangu.py \
  --model_dir /opt/pangu/openPangu-Embedded-1B-V1.1-lora-merged \
  --data_json /opt/eval_data.json \
  --out_jsonl /opt/pred.jsonl \
  --device npu --dtype bfloat16 \
  --max_new_tokens 2048 --temperature 0.0 \
  --local_files_only
```

### 4.2 评测

```bash
python pangu/eval/eval_metrics.py \
  --gold_json /opt/eval_data.json \
  --pred_jsonl /opt/pred.jsonl \
  --pred_field answer \
  --match_mode order \
  --save /opt/metrics.json
```
