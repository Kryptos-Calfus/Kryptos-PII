"""Map nvidia/Nemotron-PII onto the Kryptos taxonomy.

    python finetune/nemotron.py report          # what the map does, no download
    python finetune/nemotron.py convert --limit 20000 --out finetune/nemotron.json

200,000 rows (100K train / 100K test), CC-BY-4.0, character offsets, `us` and
`intl` locales. It fixes six of the nine misses on our held-out set: IPv4,
spelled-out dates, international phone formats, postcodes and street addresses.

It cannot be used as-is, and that is the whole reason this file exists.

**Nemotron's definition of PII is much broader than ours.** `company_name` is
its most frequent label -- more common than `first_name` -- and it also marks
occupation, employment status, education, city, state, country and language as
personal information. Our question asks whether a span is personal information
about *a specific individual*, and our own held-out set says plainly that
`Chennai Super Kings` and `Civil Lines` are not. Training on these labels
unmapped would teach the detector that `Microsoft` is PII and collapse precision
on exactly the cases we measure.

Inverted, that same breadth is the most valuable half of the dataset. Our worst
false positives are company and place names; Nemotron labels them in bulk, so
mapped to the negative side they become the hard negatives we have never had.

Four buckets, and every one of the 55 labels is in exactly one:

    ADOPT      a Kryptos type. Trains as positive.
    NEGATIVE   real, labelled, and not about an individual. Trains as negative,
               which is where most of the precision win comes from.
    SENSITIVE  GDPR Article 9 special categories. Personal but not identifying,
               so they get their own type rather than being folded into one that
               would distort it.
    SKIP       genuinely ambiguous. Dropped rather than guessed, because a
               coin-flip label teaches the model a coin flip.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import sys
import urllib.request
from pathlib import Path

DATASET = "nvidia/Nemotron-PII"
ROWS_API = "https://datasets-server.huggingface.co/rows"

SENSITIVE = "sensitive_attribute"

# --- the map -------------------------------------------------------------

ADOPT: dict[str, str] = {
    "first_name": "person_name",
    "last_name": "person_name",
    "email": "email",
    "phone_number": "phone",
    "fax_number": "phone",
    "street_address": "address",
    "postcode": "address",
    "coordinate": "address",
    "date_of_birth": "date_of_birth",
    # Adopted on request. Our taxonomy has one date type and classify() already
    # returns date_of_birth for any date it recognises, so training anything
    # else here would disagree with what inference produces. The semantics are
    # looser than the name: Nemotron's `date` is any date in a personal
    # document, not specifically a birth date.
    #
    # This is the riskiest entry in the map. It is the second-largest label
    # (369 of 4141 spans sampled) and it teaches the model that dates are
    # personal, which will also fire on invoice dates and timestamps. The
    # frozen 111-case set is what decides whether that trade is worth it.
    "date": "date_of_birth",
    # Government and institutional identifiers: issued to one person, and the
    # ones our failure analysis showed us missing or mistyping.
    "ssn": "government_id",
    "national_id": "government_id",
    "tax_id": "government_id",
    "certificate_license_number": "government_id",
    "license_plate": "government_id",
    "vehicle_identifier": "government_id",
    "medical_record_number": "government_id",
    "health_plan_beneficiary_number": "government_id",
    # Money.
    "credit_debit_card": "payment_card",
    "account_number": "financial",
    "bank_routing_number": "financial",
    "swift_bic": "financial",
    # Credentials. These must never depend on a neural vote at inference, but
    # training on them still teaches the surrounding context.
    "password": "secret",
    "pin": "secret",
    "cvv": "secret",
    "api_key": "secret",
    "http_cookie": "secret",
    "user_name": "secret",
    # Network.
    "ipv4": "network_address",
    "ipv6": "network_address",
    "mac_address": "network_address",
    # Identifiers that single out a person without naming an authority.
    "customer_id": "personal_data",
    "employee_id": "personal_data",
    "unique_id": "personal_data",
    "device_identifier": "personal_data",
    "biometric_identifier": "personal_data",
}

# Labelled by Nemotron, not personal information about an individual under our
# definition. These are the hard negatives: every one of them is a category our
# detector currently gets wrong in the false-positive direction.
NEGATIVE: dict[str, str] = {
    "company_name": "our worst FP category: 'Chennai Super Kings' scored 1.000",
    "occupation": "describes a role, not a person",
    "employment_status": "'full-time' is not an identifier",
    "education_level": "describes a qualification, not a person",
    "city": "'Civil Lines' scored 0.999 as a person name",
    "state": "a region is not an individual",
    "county": "a region is not an individual",
    "country": "a country is not an individual",
    "language": "'फोन नंबर' scored 0.961 as a person name",
}

# Personal, legally loaded, and not identifying. Folding these into person_name
# or personal_data would distort both; they get their own type.
SENSITIVE_LABELS = {
    "age",
    "gender",
    "race_ethnicity",
    "religious_belief",
    "sexuality",
    "political_view",
    "blood_type",
}

# Dropped rather than guessed.
SKIP: dict[str, str] = {
    "date_time": "a timestamp is a machine event; `date` is adopted, this is not",
    "time": "a clock time alone identifies nobody",
    "url": "a profile URL is personal, a documentation link is not; Nemotron does not distinguish",
}

ALL_LABELS = set(ADOPT) | set(NEGATIVE) | SENSITIVE_LABELS | set(SKIP)


def bucket_of(label: str) -> tuple[str, str | None]:
    """(bucket, kryptos_type). An unmapped label is a hole in the map, not a default."""
    if label in ADOPT:
        return "adopt", ADOPT[label]
    if label in NEGATIVE:
        return "negative", None
    if label in SENSITIVE_LABELS:
        return "sensitive", SENSITIVE
    if label in SKIP:
        return "skip", None
    return "unmapped", None


# --- reading the dataset -------------------------------------------------


def _spans(raw) -> list[dict]:  # noqa: ANN001 - the column arrives as a repr string
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []


def sample_rows(split: str, offsets: list[int], per: int = 100):  # noqa: ANN201
    """Pull rows over HTTP, so `report` needs no download and no datasets package."""
    for off in offsets:
        url = f"{ROWS_API}?dataset={DATASET.replace('/', '%2F')}&config=default&split={split}&offset={off}&length={per}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.load(response)
        except Exception as exc:  # noqa: BLE001 - any fetch failure is one condition
            print(f"  (offset {off} unavailable: {exc})", file=sys.stderr)
            continue
        for row in payload.get("rows", []):
            yield row["row"]


def convert_row(row: dict) -> dict | None:
    """One Nemotron row -> one training example in our format.

    Returns ``pii_spans`` as ``(start, end, type)``, which is exactly what
    ``train.py`` already consumes, plus ``keep`` for the hard negatives.
    """
    text = row.get("text")
    if not isinstance(text, str) or not text:
        return None

    positives: list[tuple[int, int, str]] = []
    negatives: list[str] = []
    source_labels: list[str] = []
    for span in _spans(row.get("spans")):
        label = span.get("label")
        value = span.get("text")
        start, end = span.get("start"), span.get("end")
        if not isinstance(value, str) or not isinstance(start, int) or not isinstance(end, int):
            continue
        # Trust the offsets only when they still point at the value.
        if text[start:end] != value:
            continue
        bucket, kryptos_type = bucket_of(label)
        if bucket in ("adopt", "sensitive"):
            positives.append((start, end, kryptos_type))
            source_labels.append(label)
        elif bucket == "negative":
            negatives.append(value)

    if not positives and not negatives:
        return None
    return {
        "text": text,
        "pii_spans": positives,
        "pii": [text[a:b] for a, b, _ in positives],
        "keep": negatives,
        "source_labels": source_labels,
        "locale": row.get("locale"),
        "source": f"nemotron:{row.get('uid', '?')}",
    }


# --- commands ------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    print(f"Sampling {DATASET} over HTTP (no download)\n")
    offsets = [0, 12_500, 25_000, 50_000, 75_000, 99_000]
    seen: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    rows = 0
    for row in sample_rows("train", offsets, per=100):
        rows += 1
        for span in _spans(row.get("spans")):
            label, value = span.get("label"), span.get("text")
            if not isinstance(label, str):
                continue
            seen[label] += 1
            if isinstance(value, str) and len(examples[label]) < 3:
                examples[label].append(value)

    print(f"{rows} rows sampled, {len(seen)} distinct labels\n")
    buckets: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for label, count in seen.most_common():
        buckets[bucket_of(label)[0]].append((label, count))

    titles = {
        "adopt": "ADOPT - trains as positive, mapped to a Kryptos type",
        "negative": "NEGATIVE - trains as a hard negative, where the precision win is",
        "sensitive": f"SENSITIVE - new '{SENSITIVE}' type",
        "skip": "SKIP - ambiguous, dropped rather than guessed",
        "unmapped": "UNMAPPED - a hole in the map, fix before converting",
    }
    for name in ("adopt", "negative", "sensitive", "skip", "unmapped"):
        entries = buckets.get(name, [])
        if not entries and name != "unmapped":
            continue
        total = sum(c for _, c in entries)
        print(f"{titles[name]}  ({len(entries)} labels, {total} spans)")
        if not entries:
            print("   none\n")
            continue
        for label, count in entries:
            target = ADOPT.get(label) or (SENSITIVE if label in SENSITIVE_LABELS else "")
            why = NEGATIVE.get(label, "") or SKIP.get(label, "")
            arrow = f"-> {target}" if target else ""
            sample = ", ".join(repr(v) for v in examples[label][:2])
            print(f"   {label:<30}{count:>5}  {arrow:<22}{sample[:46]}")
            if why:
                print(f"   {'':<30}{'':>5}  {why}")
        print()

    missing = sorted(ALL_LABELS - set(seen))
    if missing:
        print(f"mapped but not seen in this sample: {', '.join(missing)}")
    return 1 if buckets.get("unmapped") else 0


def _parquet_rows(split: str, cache: Path):  # noqa: ANN202
    """Read the split from its parquet file, fetching it once into ``cache``.

    Deliberately not ``datasets.load_dataset``: this is a build step that should
    pull one known file to one known place, so a later run is offline and the
    training pipeline never reaches the network. Nothing in the shipped package
    imports this module.
    """
    import pyarrow.parquet as pq

    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"nemotron-{split}.parquet"
    if not local.exists():
        import urllib.request

        url = (
            f"https://huggingface.co/api/datasets/{DATASET}/parquet/default/{split}/0.parquet"
        )
        print(f"fetching {url}\n     -> {local}")
        urllib.request.urlretrieve(url, local)
    size = local.stat().st_size / 1e6
    print(f"reading {local.name} ({size:.0f} MB)")

    table = pq.read_table(local)
    columns = table.column_names
    for batch in table.to_batches(max_chunksize=2000):
        rows = batch.to_pydict()
        for i in range(len(rows[columns[0]])):
            yield {c: rows[c][i] for c in columns}


def cmd_convert(args: argparse.Namespace) -> int:
    stream = _parquet_rows(args.split, Path(args.cache))

    out: list[dict] = []
    stats: collections.Counter[str] = collections.Counter()
    by_type: collections.Counter[str] = collections.Counter()
    by_label: collections.Counter[str] = collections.Counter()
    for row in stream:
        example = convert_row(row)
        if example is None:
            stats["dropped"] += 1
            continue
        out.append(example)
        stats["kept"] += 1
        stats["positives"] += len(example["pii_spans"])
        stats["negatives"] += len(example["keep"])
        for _, _, kryptos_type in example["pii_spans"]:
            by_type[kryptos_type] += 1
        for label in example["source_labels"]:
            by_label[label] += 1
        if args.limit and len(out) >= args.limit:
            break

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(
        f"\nwrote {len(out):,} examples to {args.out}\n"
        f"  positive spans {stats['positives']:,}\n"
        f"  hard negatives {stats['negatives']:,}\n"
        f"  rows dropped   {stats['dropped']:,} (no mappable span)"
    )
    print("\nPOSITIVE SPANS BY KRYPTOS TYPE")
    for kryptos_type, count in by_type.most_common():
        print(f"   {kryptos_type:<20}{count:>9,}")
    print("\nTOP SOURCE LABELS")
    for label, count in by_label.most_common(12):
        print(f"   {label:<30}{count:>9,}  -> {ADOPT.get(label) or SENSITIVE}")
    print(f"\n   'date' spans converted: {by_label.get('date', 0):,}")
    print(
        "\nThe 111-case set in finetune/test_set.json stays the regression gate: "
        "it encodes our definition, and these labels encode NVIDIA's."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nemotron", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report", help="Show what the label map does, without downloading")
    conv = sub.add_parser("convert", help="Convert the dataset into our training format")
    conv.add_argument("--split", default="train")
    conv.add_argument("--limit", type=int, default=0, help="0 means all")
    conv.add_argument("--out", default="finetune/nemotron.json")
    conv.add_argument("--cache", default="finetune/.nemotron-cache",
                      help="where the parquet is kept, so a second run is offline")
    args = parser.parse_args(argv)
    return cmd_report(args) if args.command == "report" else cmd_convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
