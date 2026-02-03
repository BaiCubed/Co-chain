from __future__ import annotations
import argparse, json, os
import numpy as np
import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.gleu_score import sentence_gleu
from rouge_chinese import Rouge

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

try:
    from bert_score import score as bert_score
    _bert_ok = True
except Exception:
    _bert_ok = False


def read_gold_list(gold_json: str):
    with open(gold_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for i, item in enumerate(data):
        ins = (item.get("instruction") or "").strip()
        ref = (item.get("output") or "").strip()
        out.append((i, ins, ref))
    return out


def read_gold_map(gold_json: str) -> dict:
    gold_list = read_gold_list(gold_json)
    gold = {}
    for _, ins, ref in gold_list:
        if ins and ref:
            gold[ins] = ref
    return gold


def read_pred(pred_jsonl: str, pred_field: str, pred_idx_field: str):
    rows = []
    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            ins = (obj.get("instruction") or "").strip()
            pred = (obj.get(pred_field) or "").strip()
            idx = obj.get(pred_idx_field, None)
            rows.append({"line_no": line_no, "idx": idx, "instruction": ins, "pred": pred})
    return rows


def tokenize_zh(s: str):
    return list(jieba.cut(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_json", required=True, help="JSON list with instruction/output")
    ap.add_argument("--pred_jsonl", required=True, help="JSONL with instruction + pred_field")
    ap.add_argument("--pred_field", default="output")
    ap.add_argument("--save", default="", help="save aggregated metrics json")
    ap.add_argument("--save_per_sample", default="", help="optional: save per-sample jsonl")

    ap.add_argument("--match_mode", default="instruction", choices=["instruction", "order", "idx"],
                    help="How to align pred with gold. order/idx ignores instruction.")
    ap.add_argument("--pred_idx_field", default="idx", help="pred idx field name for match_mode=idx")

    args = ap.parse_args()

    gold_list = read_gold_list(args.gold_json)
    gold_map = read_gold_map(args.gold_json)
    pred_rows = read_pred(args.pred_jsonl, args.pred_field, args.pred_idx_field)

    rouge = Rouge()
    smooth = SmoothingFunction().method1

    score_dict = {k: [] for k in ["bleu1","bleu2","bleu4","gleu","meteor","rouge-1","rouge-2","rouge-l"]}
    bert_preds, bert_labels = [], []
    per_out = []

    def add_one(ins_for_log: str, pred: str, label: str):
        hyp = tokenize_zh(pred)
        ref = tokenize_zh(label)

        score_dict["bleu1"].append(sentence_bleu([ref], hyp, weights=(1,0,0,0), smoothing_function=smooth)*100)
        score_dict["bleu2"].append(sentence_bleu([ref], hyp, weights=(0.5,0.5,0,0), smoothing_function=smooth)*100)
        score_dict["bleu4"].append(sentence_bleu([ref], hyp, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)*100)
        score_dict["gleu"].append(sentence_gleu([ref], hyp)*100)

        try:
            score_dict["meteor"].append(meteor_score([" ".join(ref)], " ".join(hyp))*100)
        except Exception:
            score_dict["meteor"].append(0.0)

        try:
            scores = rouge.get_scores(" ".join(hyp), " ".join(ref))[0]
            score_dict["rouge-1"].append(scores["rouge-1"]["f"]*100)
            score_dict["rouge-2"].append(scores["rouge-2"]["f"]*100)
            score_dict["rouge-l"].append(scores["rouge-l"]["f"]*100)
        except Exception:
            score_dict["rouge-1"].append(0.0)
            score_dict["rouge-2"].append(0.0)
            score_dict["rouge-l"].append(0.0)

        if _bert_ok:
            bert_preds.append(pred)
            bert_labels.append(label)

        if args.save_per_sample:
            per_out.append({"instruction": ins_for_log, "pred": pred, "label": label})

    n_matched = 0

    if args.match_mode == "instruction":
        for row in pred_rows:
            ins = (row["instruction"] or "").strip()
            pred = (row["pred"] or "").strip()
            if not ins or ins not in gold_map:
                continue
            label = gold_map[ins]
            add_one(ins_for_log=ins, pred=pred, label=label)
            n_matched += 1

    elif args.match_mode == "order":
        n = min(len(pred_rows), len(gold_list))
        for i in range(n):
            _, gold_ins, label = gold_list[i]
            pred = (pred_rows[i]["pred"] or "").strip()
            add_one(ins_for_log=gold_ins, pred=pred, label=(label or ""))
            n_matched += 1

    elif args.match_mode == "idx":
        gold_by_idx = {i: (ins, ref) for i, ins, ref in gold_list}
        for row in pred_rows:
            idx = row.get("idx", None)
            pred = (row["pred"] or "").strip()
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if idx_int not in gold_by_idx:
                continue
            gold_ins, label = gold_by_idx[idx_int]
            add_one(ins_for_log=gold_ins, pred=pred, label=(label or ""))
            n_matched += 1

    agg = {k: float(np.mean(v)) if v else 0.0 for k, v in score_dict.items()}
    agg["n_matched"] = int(n_matched)

    if _bert_ok and bert_preds:
        P, R, F1 = bert_score(bert_preds, bert_labels, lang="zh", verbose=False)
        agg["bert_p"] = float(P.mean().item()*100)
        agg["bert_r"] = float(R.mean().item()*100)
        agg["bert_f1"] = float(F1.mean().item()*100)

    print(json.dumps(agg, ensure_ascii=False, indent=2))

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)

    if args.save_per_sample:
        os.makedirs(os.path.dirname(args.save_per_sample) or ".", exist_ok=True)
        with open(args.save_per_sample, "w", encoding="utf-8") as f:
            for it in per_out:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
