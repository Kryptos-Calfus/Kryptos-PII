"""Metrics and error analysis shared by train.py and compare_current.py.

Three levels, reported separately because they answer different questions:

  CANDIDATE  does every gold PII value reach LAYA at all? A value the extractor never
             proposes cannot be recovered by fine-tuning, so this is a ceiling, not a score.
  PIECE      of the candidates LAYA did judge, how many did it judge correctly?
             Only meaningful for candidate-based engines.
  VALUE      did the labelled PII disappear from the masked text, and did the labelled
             clean text survive? This is the only level comparable across every engine,
             including ones that never expose candidates (Presidio, GPT, the regex engine).
"""

import json
from pathlib import Path

from common import pieces, spans_of


def prob(score):
    """A scorer returns either P(PII) or (P(PII), type); this reads the probability."""
    return score[0] if isinstance(score, tuple) else score


def kind(score):
    """The placeholder to write: the predicted type, or PII when the model is the binary one."""
    return score[1].upper() if isinstance(score, tuple) else "PII"

ROOT = Path(__file__).resolve().parents[1]


def load_split():
    """(train, test). Strict and unchanged: even data.json -> train, odd -> test, HOLDOUT -> test."""
    import sys
    sys.path[:0] = [str(ROOT / "evals")]
    from holdout import HOLDOUT
    from labels import LABELS

    texts = json.loads((ROOT / "finetune" / "data.json").read_text())
    real = [{"text": t, "pii": l["pii"], "keep": l["keep"]} for t, l in zip(texts, LABELS)]
    return real[0::2], real[1::2] + [dict(h) for h in HOLDOUT]


def gold_spans(x):
    """(start, end) pairs; the type carried by typed gold spans is not needed for scoring."""
    return [(a, b) for a, b, *_ in spans_of(x["text"], x["pii"])]


# ---------------------------------------------------------------- candidate level

def candidate_recall(dataset, examples=20):
    """Does at least one candidate overlap each gold PII value?"""
    total = covered = 0
    missed = []
    for x in dataset:
        ps = pieces(x["text"])
        for v in x["pii"]:
            total += 1
            sp = spans_of(x["text"], [v])
            if sp and any(s < b and a < e for a, b in sp for s, e in ps):
                covered += 1
            else:
                missed.append({"text": x["text"], "value": v,
                               "why": "label not found in text" if not sp else "no candidate overlaps it"})
    return {"gold": total, "covered": covered, "missed": missed[:examples],
            "n_missed": len(missed), "recall": covered / total if total else 1.0}


# ---------------------------------------------------------------- scoring engines

def score_dataset(dataset, score_fn):
    """Run a per-candidate scorer once over the dataset; thresholds are applied later.

    score_fn(text, spans) -> list of P(PII), aligned with spans.
    """
    rows = []
    for x in dataset:
        ps = pieces(x["text"])
        rows.append({"x": x, "spans": ps, "probs": score_fn(x["text"], ps), "gold": gold_spans(x)})
    return rows


def mask_with(row, threshold):
    text = row["x"]["text"]
    out, cur = [], 0
    for (s, e), p in zip(row["spans"], row["probs"]):
        if prob(p) >= threshold:
            out.append(text[cur:s]); out.append(f"[{kind(p)}]"); cur = e
    out.append(text[cur:])
    return "".join(out)


def piece_metrics(rows, threshold):
    tp = fp = fn = tn = 0
    for r in rows:
        for (s, e), p in zip(r["spans"], r["probs"]):
            y = any(s < b and a < e for a, b in r["gold"])
            pred = prob(p) >= threshold
            tp += pred and y; fp += pred and not y; fn += (not pred) and y; tn += (not pred) and not y
    return _prf(tp, fp, fn, tn)


def value_metrics(dataset, masked_texts):
    """Comparable across engines: did each labelled value disappear / survive as it should?"""
    tp = fp = fn = tn = 0
    touched = clean_inputs = 0
    for x, masked in zip(dataset, masked_texts):
        low = masked.lower()
        for v in x["pii"]:
            if v.lower() in low:
                fn += 1
            else:
                tp += 1
        for k in x["keep"]:
            if k.lower() in low:
                tn += 1
            else:
                fp += 1
        if not x["pii"]:
            clean_inputs += 1
            touched += masked != x["text"]
    m = _prf(tp, fp, fn, tn)
    m.update(clean_inputs_touched=touched, clean_inputs=clean_inputs,
             pii_caught=tp, pii_total=tp + fn, clean_kept=tn, clean_total=tn + fp)
    return m


def _prf(tp, fp, fn, tn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


# ---------------------------------------------------------------- reporting

def threshold_sweep(rows, thresholds=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)):
    out = []
    for t in thresholds:
        pm = piece_metrics(rows, t)
        vm = value_metrics([r["x"] for r in rows], [mask_with(r, t) for r in rows])
        out.append({"threshold": t, "piece": pm, "value": vm})
    return out


def print_sweep(sweep, chosen=None):
    print("\n[THRESHOLD SWEEP]  (piece-level = model decisions, value-level = masking outcome)")
    print(f"  {'thr':>5} | {'P':>6} {'R':>6} {'F1':>6} (piece) | {'PII caught':>12} {'clean kept':>12} "
          f"{'F1':>6} (value) | touched")
    for s in sweep:
        p, v = s["piece"], s["value"]
        mark = " <- selected" if chosen is not None and abs(s["threshold"] - chosen) < 1e-9 else ""
        print(f"  {s['threshold']:5.2f} | {p['precision']:6.1%} {p['recall']:6.1%} {p['f1']:6.3f} | "
              f"{v['pii_caught']:5}/{v['pii_total']:<6} {v['clean_kept']:5}/{v['clean_total']:<6} "
              f"{v['f1']:6.3f} | {v['clean_inputs_touched']}/{v['clean_inputs']}{mark}")


def print_report(label, rows, threshold, cand=None):
    pm = piece_metrics(rows, threshold)
    masked = [mask_with(r, threshold) for r in rows]
    vm = value_metrics([r["x"] for r in rows], masked)
    print(f"\n[{label}]  threshold={threshold}")
    if cand:
        print(f"  candidate recall     {cand['recall']:7.1%}  ({cand['covered']}/{cand['gold']} gold values reach LAYA)")
    print(f"  piece  P/R/F1        {pm['precision']:7.1%} {pm['recall']:7.1%} {pm['f1']:7.3f}"
          f"   TP {pm['tp']} FP {pm['fp']} FN {pm['fn']} TN {pm['tn']}")
    print(f"  value  P/R/F1        {vm['precision']:7.1%} {vm['recall']:7.1%} {vm['f1']:7.3f}")
    print(f"  PII caught           {vm['pii_caught']}/{vm['pii_total']} ({vm['recall']:.1%})")
    print(f"  clean values kept    {vm['clean_kept']}/{vm['clean_total']} "
          f"({vm['clean_kept'] / max(1, vm['clean_total']):.1%})")
    print(f"  clean inputs touched {vm['clean_inputs_touched']}/{vm['clean_inputs']}")
    return {"piece": pm, "value": vm}


def error_analysis(rows, threshold, cand, limit=20):
    """Separate what the extractor failed to propose from what the model judged wrongly."""
    fns, fps = [], []
    for r in rows:
        text = r["x"]["text"]
        for (s, e), p in zip(r["spans"], r["probs"]):
            y = any(s < b and a < e for a, b in r["gold"])
            if y and prob(p) < threshold:
                fns.append((text, text[s:e], prob(p)))
            elif not y and prob(p) >= threshold:
                fps.append((text, text[s:e], prob(p)))

    print(f"\n[CANDIDATE MISSES - REGEX/CANDIDATE FAILURE]  {cand['n_missed']} total "
          f"(fine-tuning LAYA cannot fix these)")
    for m in cand["missed"][:limit]:
        print(f"  gold {m['value']!r}  ({m['why']})\n       in: {m['text'][:100]!r}")
    if not cand["missed"]:
        print("  none")

    print(f"\n[FALSE NEGATIVES - LAYA CLASSIFICATION FAILURE]  {len(fns)} total "
          f"(candidate reached LAYA, LAYA said not-PII)")
    for text, span, p in sorted(fns, key=lambda z: -z[2])[:limit]:
        print(f"  span {span!r}  P(PII)={p:.3f}\n       in: {text[:100]!r}")
    if not fns:
        print("  none")

    print(f"\n[FALSE POSITIVES - LAYA CLASSIFICATION FAILURE]  {len(fps)} total "
          f"(not PII, masked anyway)")
    for text, span, p in sorted(fps, key=lambda z: -z[2])[:limit]:
        print(f"  span {span!r}  P(PII)={p:.3f}\n       in: {text[:100]!r}")
    if not fps:
        print("  none")
    return {"false_negatives": fns, "false_positives": fps}
