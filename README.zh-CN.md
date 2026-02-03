# Cochain

本仓库包含三部分：

- **Cochain（MSCoRe）**：多阶段/工作流协作推理框架（`./Cochain/`）
- **Auto-SLURP**：智能助理式多智能体评测（`./Auto-SLURP/`）
- **openPangu（Embedded-1B）**：昇腾 NPU 推理 / LoRA 微调 / 合并与复现流程（`./pangu/`）

## 目录结构

```text
.
├── Cochain/
├── Auto-SLURP/
├── pangu/
│   ├── inference/
│   ├── scripts/
│   └── eval/
├── docs/
├── requirements.txt
├── THIRD_PARTY_NOTICES.md
└── .gitignore
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

- MSCoRe：`cd Cochain/main && python main.py`
- Auto-SLURP：先 `cd Auto-SLURP/server && sh run.sh`，再 `cd ../Cochain && python test_cochain_on_autoslurp.py`

## openPangu（昇腾 NPU）

完整推理/微调/合并/复现流程见 `pangu/README.zh-CN.md`（英文版：`pangu/README.md`）。

## 第三方与许可证

- openPangu 模型以其官方仓库的 `LICENSE` 为准。
- 本仓库不分发模型权重，仅提供脚本与说明。

## 复现 openPangu 结果（合并后的模型）

完整的端到端步骤（微调/合并/推理/评测）见：`pangu/README.zh-CN.md`。

