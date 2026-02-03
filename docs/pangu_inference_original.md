# Pangu Embedded Inference Notes (Ascend NPU)

This note describes how to run a **local inference sanity-check** for **openPangu-Embedded** on **Ascend NPU**.

## 1. What this does

- Loads an openPangu-Embedded model (1B or 7B)
- Prints basic NPU / runtime information
- Prints the model architecture (`print(model)`)
- Runs a short generation example
- Saves the full stdout/stderr to a log file

## 2. Environment

Activate your environment and source Ascend runtime variables (adjust paths to your system):

```bash
conda activate pangu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## 3. Files

In this repository, the runnable entry is:

- `pangu/inference/run_pangu.py`
- `pangu/inference/run_pangu.sh`

## 4. Run

```bash
cd pangu/inference
bash run_pangu.sh /opt/pangu/openPangu-Embedded-1B-V1.1
```

To use 7B, change the model directory argument accordingly.

The script writes `run_pangu_full.log` in the same folder.

---

中文说明：见 `docs/pangu_inference_original.zh-CN.md`。
