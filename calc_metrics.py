import os
import re
import json
import csv
import torch
from pathlib import Path
from typing import Dict
import hydra


from src.metrics.utils import levenshtein


def normalize_text(s):
    s = s.strip().lower()
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff")
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_text_file(path):
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def compute_for_pair(gt_text, pred_text, normalize= True):
    if normalize:
        gt = normalize_text(gt_text)
        pred = normalize_text(pred_text)
    else:
        gt = gt_text.strip()
        pred = pred_text.strip()

    gt_words = gt.split() if gt else []
    pred_words = pred.split() if pred else []
    word_edits = levenshtein(gt_words, pred_words)
    n_words = len(gt_words)

    gt_chars = list(gt) if gt else []
    pred_chars = list(pred) if pred else []
    char_edits = levenshtein(gt_chars, pred_chars)
    n_chars = len(gt_chars)

    wer = (word_edits / n_words) if n_words > 0 else 0.0
    cer = (char_edits / n_chars) if n_chars > 0 else 0.0

    return {
        "word_edits": word_edits,
        "n_words": n_words,
        "wer": wer,
        "char_edits": char_edits,
        "n_chars": n_chars,
        "cer": cer,
    }


import re
from pathlib import Path

def find_txt_files(dirpath):
    file_map = {}
    id_pattern = re.compile(r"ID.*")
    for p in dirpath.glob("*.txt"):
        if p.is_file():
            match = id_pattern.search(p.name)
            if match:
                file_map[match.group(0)] = p
    return file_map

@hydra.main(version_base=None, config_path="src/configs", config_name="metrics_eval")
def main(cfg):
    gt_dir = Path(cfg.paths.gt_dir)
    pred_dir = Path(cfg.paths.pred_dir)
    normalize = cfg.get("normalize", True)
    include_extras = cfg.get("include_extras", False)
    out_path = Path(cfg.get("out_path", "metrics_report.json"))
    min_reported = int(cfg.get("min_reported", 10))

    gt_map = find_txt_files(gt_dir)
    pred_map = find_txt_files(pred_dir)

    total_word_edits = total_words = 0
    total_char_edits = total_chars = 0

    results = []
    missing_preds, missing_gt = [], []

    for utt_id, gt_path in gt_map.items():
        gt_text = read_text_file(gt_path)
        pred_path = pred_map.get(utt_id)
        pred_text = read_text_file(pred_path) if pred_path else ""
        if pred_path is None:
            missing_preds.append(utt_id)

        metrics = compute_for_pair(gt_text, pred_text, normalize)
        results.append({"id": utt_id, **metrics})
        total_word_edits += metrics["word_edits"]
        total_words += metrics["n_words"]
        total_char_edits += metrics["char_edits"]
        total_chars += metrics["n_chars"]

 

    overall_wer = total_word_edits / total_words if total_words > 0 else 0
    overall_cer = total_char_edits / total_chars if total_chars > 0 else 0

    print(f"Evaluated {len(results)} samples")
    print(f"WER = {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print(f"CER = {overall_cer:.4f} ({overall_cer*100:.2f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "wer": overall_wer,
            "cer": overall_cer,
            "n_utts": len(results),
            "total_words": total_words,
            "total_chars": total_chars,
        },
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON report to {out_path}")

if __name__ == "__main__":
    main()
