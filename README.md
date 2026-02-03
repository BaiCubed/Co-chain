# Cochain

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-4.x-brightgreen)](https://neo4j.com/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-supported-orange)](https://github.com/huggingface/transformers)

This repository bundles three components:

- **Cochain (MSCoRe benchmark)**: workflow-style collaborative reasoning (`./Cochain/`)
- **Auto-SLURP benchmark**: assistant-style multi-agent evaluation (`./Auto-SLURP/`)
- **openPangu (Embedded-1B)**: Ascend NPU inference / LoRA fine-tuning / merge utilities (`./pangu/`)

> Chinese docs: see `README.zh-CN.md`.

---

## Repository layout

```text
.
├── Cochain/                     # Cochain (MSCoRe)
├── Auto-SLURP/                  # Auto-SLURP benchmark
├── pangu/                       # openPangu on Ascend NPU
│   ├── inference/               # inference scripts
│   ├── scripts/                 # LoRA fine-tuning + merge
│   └── eval/                    # scoring
├── docs/                        # notes
├── requirements.txt
├── THIRD_PARTY_NOTICES.md
└── .gitignore
```

---

## Installation

```bash
pip install -r requirements.txt
```

For scoring (ROUGE/METEOR/BERTScore), the required packages are already listed in `requirements.txt`.

---

## Run: MSCoRe (Cochain)

```bash
cd Cochain/main
python main.py
```

---

## Run: Auto-SLURP

Start the mock server:

```bash
cd Auto-SLURP/server
sh run.sh
```

Run the Cochain adapter:

```bash
cd ../Cochain
python test_cochain_on_autoslurp.py
```

---

## openPangu on Ascend NPU

See `pangu/README.md` for:

- Base-model inference sanity-check
- LoRA fine-tuning (torchrun + HCCL)
- Merging LoRA into the base model
- Reproducing evaluation (merged model -> predictions -> scores)

---

## Third-party & license

- openPangu models are released under the **OPENPANGU MODEL LICENSE AGREEMENT (Version 1.0)** (see the `LICENSE` file in the model repository you download).
- This repository **does not distribute** openPangu weights; it only provides scripts and documentation.

More details: `THIRD_PARTY_NOTICES.md`.

## Reproduce openPangu results (merged model)

A minimal end-to-end example (merge -> predict -> score) is documented in `pangu/README.md`.

Quick commands:

```bash
# 1) merge LoRA -> base (CPU merge)
python pangu/scripts/merge_lora_to_model.py   --model_dir /opt/pangu/openPangu-Embedded-1B-V1.1   --adapter_dir /opt/pangu_demo/pangu_lora_adapter   --out_dir /opt/pangu/openPangu-Embedded-1B-V1.1-lora-merged   --dtype bfloat16 --device cpu

# 2) run merged model -> JSONL predictions
ASCEND_RT_VISIBLE_DEVICES=0 python pangu/inference/run_merged_pangu.py   --model_dir /opt/pangu/openPangu-Embedded-1B-V1.1-lora-merged   --data_json /opt/eval_data.json   --out_jsonl /opt/pred.jsonl   --device npu --dtype bfloat16   --max_new_tokens 2048 --temperature 0.0 --local_files_only

# 3) score
python pangu/eval/eval_metrics.py   --gold_json /opt/eval_data.json   --pred_jsonl /opt/pred.jsonl   --pred_field answer   --match_mode order   --save /opt/metrics.json
```

