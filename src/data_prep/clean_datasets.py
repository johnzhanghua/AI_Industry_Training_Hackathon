#!/usr/bin/env python3
"""Clean RBA/ASX/AFR raw data into a training-only corpus for Nemotron fine-tuning.

Read-only against 'data_set2/raw/'. Never writes there. Output goes to data_set2/cleaned/.
This cleaned copy is for the FINE-TUNING PIPELINE ONLY. The runtime query_data tool
(G3's job, src/) must keep reading the raw files at full precision -- grading tolerance
on ASX closes is +/-0.0001, rounding here would blow past that if reused at runtime.

See TEAM_PLAN.md phase 1 (G1) and Sonali_plan.md for the reasoning behind these steps.

Usage:
    python src/data_prep/clean_datasets.py
    python src/data_prep/clean_datasets.py --out-dir data_set2 --skip-validate
"""

import argparse
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data_set2" / "raw"
RAW_RBA = RAW_ROOT / "RBA Rates" / "RBA-rates.jsonl"
RAW_ASX_DIR = RAW_ROOT / "ASX"
RAW_AFR_DIR = RAW_ROOT / "AFR"

RBA_DATE_FMT_IN = "%d %b %Y"
SOFT_HYPHEN = "\xad"
NBSP = "\xa0"

# reference facts pulled from Participant_Package/public_questions.jsonl -- used to
# sanity-check that cleaning didn't shift exact counts away from the graded answer.
QBE_2021_EXPECTED_COUNT = 369  # MHQ076: whole-word QBE, 2021 only


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------- step 1: RBA
def clean_rba(clean_root: Path):
    rows = []
    for rec in read_jsonl(RAW_RBA):
        eff_date = datetime.strptime(rec["Effective Date"], RBA_DATE_FMT_IN).strftime("%Y-%m-%d")
        change_pct = float(rec["Change % points"])
        rows.append({
            "effective_date": eff_date,
            "change_pct": change_pct,
            "change_bps": round(change_pct * 100),  # x100, NOT x10000 -- check the math yourself
            "cash_rate_target_pct": float(rec["Cash rate target%"]),
        })
    out_path = clean_root / "RBA Rates" / RAW_RBA.name
    write_jsonl(out_path, rows)
    print(f"[1/4] rba clean  -> {out_path}  ({len(rows)} rows)")


# ---------------------------------------------------------------- step 2: ASX
def clean_asx(clean_root: Path):
    total, n_files = 0, 0
    for src in sorted(RAW_ASX_DIR.glob("*.jsonl")):
        rows = []
        for rec in read_jsonl(src):
            ticker = rec["ticker"]
            rows.append({
                "ticker": ticker,
                "clean_ticker": ticker.split(".")[0],
                "date": rec["date"],  # already ISO-8601
                "open": round(float(rec["open"]), 2),
                "high": round(float(rec["high"]), 2),
                "low": round(float(rec["low"]), 2),
                "close": round(float(rec["close"]), 2),
                "volume": int(rec["volume"]),
            })
        write_jsonl(clean_root / "asx" / src.name, rows)
        total += len(rows)
        n_files += 1
    print(f"[2/4] asx clean  -> {clean_root / 'asx'}  ({total} rows / {n_files} files)")


# ---------------------------------------------------------------- step 3: AFR
def clean_afr_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = text.replace(SOFT_HYPHEN, "")
    text = text.replace(NBSP, " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def clean_afr_date(raw: str) -> str:
    raw = raw or ""
    if re.fullmatch(r"\d{8}", raw):
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw


def clean_afr(clean_root: Path):
    total, n_files = 0, 0
    for src in sorted(RAW_AFR_DIR.glob("AFR_*.jsonl")):
        rows = []
        for rec in read_jsonl(src):
            rows.append({
                "headline": clean_afr_text(rec.get("HEADLINE")),
                "subhead": clean_afr_text(rec.get("SUBHEAD")),
                "intro": clean_afr_text(rec.get("INTRO")),
                "text": clean_afr_text(rec.get("TEXT")),
                "newspaper": rec.get("NEWSPAPER", ""),
                "publication_date": clean_afr_date(rec.get("PUBLICATIONDATE")),
            })
        write_jsonl(clean_root / "afr" / src.name, rows)
        total += len(rows)
        n_files += 1
    print(f"[3/4] afr clean  -> {clean_root / 'afr'}  ({total} rows / {n_files} files)")


# ---------------------------------------------------- step 4: validate counts didn't shift
def count_pattern(dir_path, field_names, pattern, year=None):
    rx = re.compile(pattern, re.IGNORECASE)
    count = 0
    for f in sorted(dir_path.glob("AFR_*.jsonl")):
        if year and not f.name.startswith(f"AFR_{year}"):
            continue
        for rec in read_jsonl(f):
            combined = " ".join(rec.get(k, "") or "" for k in field_names)
            if rx.search(combined):
                count += 1
    return count


def validate(clean_root: Path):
    raw_fields = ["HEADLINE", "SUBHEAD", "INTRO", "TEXT"]
    clean_fields = ["headline", "subhead", "intro", "text"]
    pattern = r"\bQBE\b"

    raw_count = count_pattern(RAW_AFR_DIR, raw_fields, pattern, year="2021")
    clean_count = count_pattern(clean_root / "afr", clean_fields, pattern, year="2021")

    print(f"[4/4] validate   -> QBE 2021 whole-word count: "
          f"raw={raw_count} clean={clean_count} expected={QBE_2021_EXPECTED_COUNT}")

    if raw_count != QBE_2021_EXPECTED_COUNT:
        print("      !! raw count != published reference. Counting method itself needs a look.")
    if clean_count != raw_count:
        print("      !! cleaning shifted the count vs raw text. "
              "Use raw text for exact pattern-count metrics; keep cleaned text for retrieval only.")
    if raw_count == QBE_2021_EXPECTED_COUNT == clean_count:
        print("      OK -- matches reference exactly, cleaned text safe for counts too.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",
                         default=str(ROOT / "data_set2"),
                         help="output root for cleaned data (default: data_set2)")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    clean_root = Path(args.out_dir) / "cleaned"
    clean_root.mkdir(parents=True, exist_ok=True)

    clean_rba(clean_root)
    clean_asx(clean_root)
    clean_afr(clean_root)

    if not args.skip_validate:
        validate(clean_root)

    print(f"\nDone. Cleaned training corpus: {clean_root}")
    print("Reminder: training-only. Runtime query_data tool reads 'data_set2/raw', full precision.")


if __name__ == "__main__":
    main()

