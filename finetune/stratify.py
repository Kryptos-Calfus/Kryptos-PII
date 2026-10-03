"""Pick a training subsample that targets what the detector actually gets wrong.

    python finetune/stratify.py --n 6000 --out finetune/nemotron_6k.json

The converted dataset is 99,963 examples and ~3M training items: six days on
this hardware. The shipped checkpoint was trained on 6,000 synthetic examples in
under two hours, so matching that budget keeps the comparison to one variable --
where the data came from, not how much of it there is. If 6K of Nemotron beats
6K of synth on the frozen 111-case set, scaling up is justified by a measurement
instead of a hope.

Selection is driven by the failure analysis rather than by chance:

* **Misses** -- ipv4/ipv6, street_address, postcode, phone_number, date_of_birth
  were the nine values the model scored below threshold. Examples carrying them
  are oversampled.
* **Hard negatives** -- company_name, city, state, occupation are the categories
  behind the measured false positives. Examples carrying them are oversampled.
* **Dates are capped.** `date` is 77,128 of 521,825 positive spans. Left alone
  it would dominate a small sample and train a detector whose loudest signal is
  "dates are personal", which is not the lesson we are trying to teach.
* **Locale is balanced**, because `intl` is where the phone formats live.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

# Labels whose examples we want more of, and why.
TARGETS_MISSED = {"ipv4", "ipv6", "street_address", "postcode", "phone_number", "date_of_birth"}
TARGETS_NEGATIVE = {"company_name", "city", "state", "county", "country", "occupation", "language"}

# `date` is adopted, but at 15% of all positive spans it would swamp a 6K
# sample. This caps how many selected examples may lean on it.
DATE_SHARE = 0.20


def score_example(example: dict) -> int:
    """How much this example exercises a known failure. Higher is picked sooner."""
    labels = set(example.get("source_labels", []))
    points = 0
    points += 3 * len(labels & TARGETS_MISSED)
    points += 2 * len(set(example.get("keep", [])) and TARGETS_NEGATIVE)
    # keep[] holds values, not labels, so weight by how many negatives are present
    points += 2 * min(len(example.get("keep", [])), 4)
    if example.get("locale") == "intl":
        points += 1
    return points


def main() -> int:
    parser = argparse.ArgumentParser(prog="stratify")
    parser.add_argument("--src", default="finetune/nemotron.json")
    parser.add_argument("--out", default="finetune/nemotron_6k.json")
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()

    data = json.loads(Path(args.src).read_text())
    print(f"source {len(data):,} examples")

    random.seed(args.seed)
    random.shuffle(data)  # break any ordering before a stable sort on score
    ranked = sorted(data, key=score_example, reverse=True)

    date_budget = int(args.n * DATE_SHARE)
    picked: list[dict] = []
    chosen: set[int] = set()
    date_heavy = 0

    # Quota pass first. Ranking alone under-serves the rare categories: at
    # n=1000 it produced 27 ipv4 and 18 ipv6 examples, and IP addresses were
    # three of the nine values the detector missed. A sample that under-
    # represents the failures cannot fix them, so each target label gets a floor
    # before general ranking fills the rest.
    floor = max(20, args.n // len(TARGETS_MISSED) // 4)
    for label in sorted(TARGETS_MISSED):
        have = 0
        for i, example in enumerate(ranked):
            if have >= floor or len(picked) >= args.n:
                break
            if i in chosen or label not in example.get("source_labels", []):
                continue
            chosen.add(i)
            picked.append(example)
            have += 1
        print(f"  quota {label:<18}{have:>5}")

    for i, example in enumerate(ranked):
        if i in chosen:
            continue
        if len(picked) >= args.n:
            break
        labels = example.get("source_labels", [])
        leans_on_date = labels.count("date") > len(labels) / 2
        if leans_on_date:
            if date_heavy >= date_budget:
                continue
            date_heavy += 1
        chosen.add(i)
        picked.append(example)

    by_type: collections.Counter[str] = collections.Counter()
    by_label: collections.Counter[str] = collections.Counter()
    locales: collections.Counter[str] = collections.Counter()
    negatives = 0
    for example in picked:
        for _, _, kryptos_type in example["pii_spans"]:
            by_type[kryptos_type] += 1
        for label in example["source_labels"]:
            by_label[label] += 1
        negatives += len(example["keep"])
        locales[example.get("locale")] += 1

    Path(args.out).write_text(json.dumps(picked, ensure_ascii=False, indent=1))
    print(f"wrote {len(picked):,} examples to {args.out}")
    print(f"  positive spans {sum(by_type.values()):,}")
    print(f"  hard negatives {negatives:,}")
    print(f"  locales        {dict(locales)}")
    print(f"  date-dominant  {date_heavy:,} (cap {date_budget:,})")
    print("\nBY KRYPTOS TYPE")
    for t, c in by_type.most_common():
        print(f"   {t:<22}{c:>8,}")
    print("\nCOVERAGE OF THE NINE MEASURED MISSES")
    for label in sorted(TARGETS_MISSED):
        print(f"   {label:<22}{by_label.get(label, 0):>8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
