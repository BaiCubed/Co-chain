# 盘古 Embedded 推理说明（Ascend NPU）

本文档用于说明如何在 **昇腾 NPU** 上对 **openPangu-Embedded** 做一次本地推理自检。

## 1. 功能

- 加载 openPangu-Embedded（1B 或 7B）
- 打印 NPU / 运行时信息
- `print(model)` 输出模型结构
- 跑一次短文本生成
- 将完整输出保存到日志文件

## 2. 环境

按你的机器路径调整：

```bash
conda activate pangu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## 3. 代码入口

- `pangu/inference/run_pangu.py`
- `pangu/inference/run_pangu.sh`

## 4. 运行

```bash
cd pangu/inference
bash run_pangu.sh /opt/pangu/openPangu-Embedded-1B-V1.1
```

脚本会在当前目录生成 `run_pangu_full.log`。
