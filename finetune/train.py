"""Fine-tune LAYA to decide, per piece of text, whether it is personal information.

    HF_HUB_OFFLINE=1 uv run python finetune/train.py                  # train + evaluate + save
    HF_HUB_OFFLINE=1 uv run python finetune/train.py --eval-only      # evaluate the saved checkpoint
    HF_HUB_OFFLINE=1 uv run python finetune/train.py --layers 8 --context 200 --epochs 3
    HF_HUB_OFFLINE=1 uv run python finetune/train.py --mine           # after training, mine hard examples

Splits (strict, unchanged):
    even-indexed data.json -> train      odd-indexed data.json -> test
    evals/holdout.py       -> test only  (never trained on, never used to pick a checkpoint)
A slice of the *training* half is held back as validation; the checkpoint with the best
validation F1 is the one saved, so the test set stays an unbiased final measurement.
"""

import argparse
import copy
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "finetune"), str(ROOT / "evals")]

import laya  # noqa: E402
from laya.common import collate_items, proper_reward  # noqa: E402

import common  # noqa: E402
from common import (NOT_PII, QUESTION, TYPE_QUESTION, TYPES, pieces, spans_of,  # noqa: E402
                    state_for, training_pieces, typed_spans_of)
from evaluate import (candidate_recall, error_analysis, load_split, piece_metrics,  # noqa: E402
                      print_report, print_sweep, score_dataset, threshold_sweep)
from synth import generate  # noqa: E402

OUT_DIR = ROOT / "finetune" / "laya-pii"
HARD_FILE = ROOT / "finetune" / "hard_examples.json"
VAL_FRACTION = 0.20   # of the training half, held back to choose the checkpoint
VAL_SYNTH = 300
MINE_POOL = 1500
SEED = 0


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=6, help="encoder layers to unfreeze (of 28): try 4, 6, 8, 12")
    p.add_argument("--context", type=int, default=120, help="context chars each side: try 60, 120, 200, 300")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--synth", type=int, default=6000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--lr-enc", type=float, default=2e-5)
    p.add_argument("--threshold", type=float, default=0.5, help="production threshold; --sweep reports alternatives")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--no-save", action="store_true",
                   help="skip writing the 1.6 GB checkpoint (sweeps only need the validation score)")
    p.add_argument("--eval-only", nargs="?", const=str(OUT_DIR), default=None)
    p.add_argument("--no-sweep", action="store_true")
    p.add_argument("--mine", action="store_true", help="after training, write hard examples for the next round")
    p.add_argument("--no-hard", action="store_true", help="ignore finetune/hard_examples.json")
    p.add_argument("--errors", type=int, default=20, help="examples printed per error category")
    p.add_argument("--types", action="store_true",
                   help="train the typed model: LAYA picks person/email/phone/... or not_pii, "
                        "so masking writes [EMAIL] instead of [PII]")
    return p.parse_args()


# ---------------------------------------------------------------- data

def datasets(args):
    """(train_texts, val_set, test_set). Test is never trained on or used for selection."""
    train_real, test = load_split()
    rng = random.Random(SEED)
    idx = list(range(len(train_real)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_FRACTION))
    val_real = [train_real[i] for i in idx[:n_val]]
    fit_real = [train_real[i] for i in idx[n_val:]]

    train = [(x["text"], x["pii_spans"]) for x in generate(args.synth, seed=SEED)]
    train += [(x["text"], typed_spans_of(x["text"], x["pii"])) for x in fit_real]

    if HARD_FILE.exists() and not args.no_hard:
        hard = json.loads(HARD_FILE.read_text())
        train += [(h["text"], [tuple(s) for s in h["pii_spans"]]) for h in hard]
        print(f"hard examples loaded from {HARD_FILE.name}: {len(hard)}")

    val = val_real + [{"text": x["text"], "pii": [x["text"][a:b] for a, b, _ in x["pii_spans"]], "keep": []}
                      for x in generate(VAL_SYNTH, seed=SEED + 7777)]
    return train, val, test


# ---------------------------------------------------------------- model plumbing

def encode(agent, text, s, e, typed=False):
    q = TYPE_QUESTION["kind"] if typed else QUESTION["pii"]
    key = "kind" if typed else "pii"
    internal = {key: agent._to_internal(q)}
    return agent._encode_state(state_for(text, s, e), [key], internal)[0]


def make_scorer(agent, batch=32, typed=False):
    """score_fn(text, spans) -> P(PII) per span. With typed=True it also returns the type."""
    k = len(TYPES) if typed else 2

    def score(text, spans):
        if not spans:
            return []
        out = []
        agent.model.eval()
        for i in range(0, len(spans), batch):
            items = [encode(agent, text, s, e, typed) for s, e in spans[i:i + batch]]
            b = collate_items([items], agent.tok.pad_token_id)
            with torch.no_grad():
                logits, _ = agent.model(*(b[k2].to(agent.device) for k2 in
                                          ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
            probs = torch.softmax(logits[:, :k].float(), -1).cpu()
            if typed:
                # P(PII) is everything that is not the not_pii option.
                not_i = TYPES.index(NOT_PII)
                for row in probs:
                    order = row.argsort(descending=True).tolist()
                    best = next(j for j in order if j != not_i)
                    out.append((1.0 - row[not_i].item(), TYPES[best]))
            else:
                out += probs[:, 1].tolist()
        return out
    return score


def save(agent, state_dict, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    base = Path(snapshot_download("convaiinnovations/laya",
                                  allow_patterns=["rl_agent_config.json", "tokenizer/*", "encoder/*"]))
    for sub in ("tokenizer", "encoder"):
        shutil.copytree(base / sub, out_dir / sub, dirs_exist_ok=True)
    cfg = dict(agent.cfg)
    cfg["temperature"] = [1.0, 1.0, 1.0]      # trained without temperature scaling
    cfg["temperature_by_options"] = {}
    cfg["pii_finetune"] = PROVENANCE
    (out_dir / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    save_file({k: v.detach().cpu().contiguous() for k, v in state_dict.items()},
              str(out_dir / "model.safetensors"))
    print(f"saved {out_dir}")


PROVENANCE = {}


# ---------------------------------------------------------------- hard-example mining

def mine(agent, args, train_real):
    """Fresh synthetic pool + the training texts, kept where the model is wrong.

    Mining never touches the test split or the holdout: those must stay unbiased.
    """
    score = make_scorer(agent, typed=args.types)
    pool = [{"text": x["text"], "pii": [x["text"][a:b] for a, b in x["pii_spans"]], "keep": []}
            for x in generate(MINE_POOL, seed=SEED + 4242)]
    pool += train_real
    hard = []
    for x in pool:
        ps = pieces(x["text"])
        if not ps:
            continue
        probs = score(x["text"], ps)
        gold = spans_of(x["text"], x["pii"])
        wrong = any((p >= args.threshold) != any(s < b and a < e for a, b in gold)
                    for (s, e), p in zip(ps, probs))
        if wrong:
            hard.append({"text": x["text"], "pii_spans": [list(g) for g in gold]})
    HARD_FILE.write_text(json.dumps(hard, indent=1, ensure_ascii=False))
    print(f"\n[MINING] {len(hard)} hard texts written to {HARD_FILE.name} "
          f"(from a fresh pool of {len(pool)}; test and holdout untouched). "
          f"Re-run train.py to fold them in.")


# ---------------------------------------------------------------- main

def main():
    args = build_args()
    common.set_context_chars(args.context)
    torch.manual_seed(SEED); random.seed(SEED)
    train, val, test = datasets(args)

    cand_test = candidate_recall(test, examples=args.errors)
    cand_val = candidate_recall(val)

    if args.eval_only:
        agent = laya.load(args.eval_only)
        rows = score_dataset(test, make_scorer(agent, typed=args.types))
        print_report(f"fine-tuned LAYA ({Path(args.eval_only).name}) on TEST", rows, args.threshold, cand_test)
        if not args.no_sweep:
            print_sweep(threshold_sweep(rows), chosen=args.threshold)
        error_analysis(rows, args.threshold, cand_test, limit=args.errors)
        return

    agent = laya.load("convaiinnovations/laya")
    print(f"\ntrain texts {len(train)} | val texts {len(val)} | test texts {len(test)} (never trained on)")
    print(f"config: layers={args.layers} context={args.context} epochs={args.epochs} synth={args.synth}")
    print(f"\n[CANDIDATE RECALL - TEST]\n  Gold PII spans: {cand_test['gold']}\n  Covered: {cand_test['covered']}"
          f"\n  Missed: {cand_test['n_missed']}\n  Recall: {cand_test['recall']:.1%}")

    base_rows = score_dataset(test, make_scorer(agent, typed=args.types))
    print_report("BASELINE base LAYA (no fine-tuning) on TEST", base_rows, args.threshold, cand_test)

    # Training candidates = regex proposals + any gold span the regex missed, so a blind spot
    # in the extractor never becomes a blind spot in the model.
    items, n_pos, n_added = [], 0, 0
    for text, gold in train:
        regex_n = len(pieces(text))
        tp = training_pieces(text, gold)
        n_added += len(tp) - regex_n
        for s, e, lab in tp:
            it = encode(agent, text, s, e, args.types)
            if args.types:
                target = [0.0] * len(TYPES)
                target[TYPES.index(lab if lab in TYPES else "person")] = 1.0
            else:
                target = [1.0 - (lab != NOT_PII), float(lab != NOT_PII)]
            it["target"] = target
            items.append(it); n_pos += lab != NOT_PII
    print(f"training pieces: {len(items)} ({n_pos} PII, {len(items) - n_pos} not PII; "
          f"{n_added} gold spans the regex alone would have missed)")

    model = agent.model
    for p in model.parameters():
        p.requires_grad = False
    enc_train = [p for layer in model.encoder.layers[-args.layers:] for p in layer.parameters()]
    enc_train += list(model.encoder.final_norm.parameters())
    head_train = [p for m in (model.head, model.type_emb, model.scorer) for p in m.parameters()]
    for p in enc_train + head_train:
        p.requires_grad = True
    opt = torch.optim.AdamW([{"params": head_train, "lr": args.lr_head},
                             {"params": enc_train, "lr": args.lr_enc}], weight_decay=0.01)
    steps = args.epochs * ((len(items) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_head, args.lr_enc],
                                                total_steps=steps, pct_start=0.06)
    print(f"trainable params: {sum(p.numel() for p in enc_train + head_train) / 1e6:.0f}M | steps {steps}")

    best = {"f1": -1.0, "epoch": None, "state": None}
    step, t0 = 0, time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(items)
        model.train()
        run, run_n = 0.0, 0
        for i in range(0, len(items), args.batch):
            b = collate_items([items[i:i + args.batch]], agent.tok.pad_token_id)
            dev = {k: b[k].to(agent.device) for k in
                   ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "target")}
            logits, _ = model(dev["input_ids"], dev["attention_mask"], dev["marker_pos"],
                              dev["marker_mask"], dev["qtype"])
            loss = -proper_reward(torch.softmax(logits, -1), dev["target"], dev["qtype"],
                                  dev["marker_mask"].float()).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_train + head_train, 1.0)
            opt.step(); sched.step(); step += 1
            run += loss.item(); run_n += 1
            if step % 200 == 0:
                el = time.perf_counter() - t0
                print(f"  epoch {epoch} step {step}/{steps} loss {run / run_n:.4f} | "
                      f"{el / 60:.1f} min elapsed, ~{el / step * (steps - step) / 60:.1f} min left", flush=True)
                run, run_n = 0.0, 0

        val_rows = score_dataset(val, make_scorer(agent, typed=args.types))
        vm = piece_metrics(val_rows, args.threshold)
        print(f"  [epoch {epoch}] VALIDATION piece F1 {vm['f1']:.4f} "
              f"(P {vm['precision']:.1%} R {vm['recall']:.1%})", flush=True)
        if vm["f1"] > best["f1"]:
            best = {"f1": vm["f1"], "epoch": epoch,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            print(f"  [epoch {epoch}] best so far, checkpoint kept")

    print(f"\nselected checkpoint: epoch {best['epoch']} (validation piece F1 {best['f1']:.4f})")
    model.load_state_dict(best["state"])
    model.to(agent.device).eval()

    PROVENANCE.update(synth=args.synth, layers=args.layers, context_chars=args.context,
                      epochs=args.epochs, selected_epoch=best["epoch"], val_f1=round(best["f1"], 4),
                      threshold=args.threshold, typed=args.types,
                      types=TYPES if args.types else None,
                      question=(TYPE_QUESTION["kind"] if args.types else QUESTION["pii"])["instructions"],
                      hard_examples=HARD_FILE.exists() and not args.no_hard)

    rows = score_dataset(test, make_scorer(agent, typed=args.types))
    print_report("FINE-TUNED LAYA on TEST", rows, args.threshold, cand_test)
    if not args.no_sweep:
        print_sweep(threshold_sweep(rows), chosen=args.threshold)
    error_analysis(rows, args.threshold, cand_test, limit=args.errors)

    if args.no_save:
        print("\n--no-save: checkpoint not written")
    else:
        save(agent, best["state"], args.out)
    if args.mine:
        mine(agent, args, [x for x in load_split()[0]])
    print(f"\n(validation candidate recall {cand_val['recall']:.1%}; selection never saw the test split)")


if __name__ == "__main__":
    main()
