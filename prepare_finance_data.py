#!/usr/bin/env python3
"""Build train/val/test pairs for LoRA fine-tuning from the AFR / ASX / RBA corpus.

Called by `03-fine-tuning.py --stage data`; also runnable on its own:

    python3 prepare_finance_data.py --data-root "data set" --out-dir data

Every pair mirrors the prompt the fine-tuned model actually sees at inference
time. `synthesis_node` in src/agent.py sends:

    Context Blocks:
    <one JSON block per executed tool call>

    Question: <the user's question>

so the training prompts are built the same way, with context blocks that use the
same keys the tools in src/agent.py emit (`query_asx_prices`,
`query_rba_cash_rate`, `query_afr_news`). The completion is the concise grounded
answer — every figure copied from the context, never inferred.

Splits are assigned by hashing a *group* key (an article, a ticker-year, an RBA
year) rather than the example, so no source record can appear on both sides of
the split and inflate the validation numbers.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

# Prompt + completion must fit MAX_SEQ_LEN 512 with room to spare, otherwise
# `truncation: longest_first` eats the answer the loss mask is computed over.
MAX_CHARS = 1600
AFR_TEXT_CHARS = 700

SPLIT_WEIGHTS = [("train", 0.8), ("val", 0.1), ("test", 0.1)]

# Articles that carry no reportable content — they poison the headline task.
AFR_SKIP = re.compile(r"crossword|quick quiz|cryptic|sudoku", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_dir(root: Path, name: str) -> Path | None:
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.lower().startswith(name.lower()):
            return child
    return None


def load_afr(root: Path) -> list[dict]:
    directory = find_dir(root, "afr")
    if directory is None:
        return []
    articles = []
    for path in sorted(directory.glob("*.jsonl")):
        for row in read_jsonl(path):
            headline = (row.get("headline") or "").strip()
            intro = (row.get("intro") or "").strip()
            text = (row.get("text") or "").strip()
            date = (row.get("publication_date") or "").strip()
            if not (headline and text and date):
                continue
            if len(text) < 200 or AFR_SKIP.search(headline):
                continue
            articles.append(
                {"headline": headline, "intro": intro, "text": text, "date": date}
            )
    return articles


def load_asx(root: Path) -> dict[str, list[dict]]:
    directory = find_dir(root, "asx")
    if directory is None:
        return {}
    by_ticker: dict[str, list[dict]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        rows = [r for r in read_jsonl(path) if r.get("date") and r.get("close") is not None]
        if not rows:
            continue
        rows.sort(key=lambda r: r["date"])
        ticker = rows[0].get("clean_ticker") or path.stem.split("-")[0].upper()
        by_ticker[ticker.upper()] = [
            {
                "ticker": ticker.upper(),
                "date": r["date"],
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": int(r["volume"]),
            }
            for r in rows
        ]
    return by_ticker


def load_rba(root: Path) -> list[dict]:
    directory = find_dir(root, "rba")
    candidates = sorted(directory.glob("*.jsonl")) if directory else []
    if not candidates:
        candidates = [Path(p) for p in glob.glob(str(root / "**" / "*rba*.jsonl"), recursive=True)]
    if not candidates:
        return []
    rows = [r for r in read_jsonl(candidates[0]) if r.get("effective_date")]
    rows.sort(key=lambda r: r["effective_date"])
    # Rename to the keys `query_rba_cash_rate` returns, so the context blocks in
    # training match the context blocks at inference.
    return [
        {
            "effective_date": r["effective_date"],
            "cash_rate_target": float(r.get("cash_rate_target_pct", r.get("cash_rate_target", 0.0))),
            "change_pct": float(r.get("change_pct", 0.0)),
            "change_bps": int(r.get("change_bps", 0)),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def make_pair(question: str, blocks: list, answer: str, group: str) -> dict | None:
    """Assemble one example in the agent's synthesis format, or None if oversized."""
    context = "\n".join(
        block if isinstance(block, str) else json.dumps(block, ensure_ascii=False)
        for block in blocks
    )
    prompt = f"Context Blocks:\n{context}\n\nQuestion: {question}"
    answer = answer.strip()
    if not answer or len(prompt) + len(answer) > MAX_CHARS:
        return None
    return {"prompt": prompt, "completion": answer, "group": group}


def money(value: float) -> str:
    return f"${value:.2f}"


def pct_change(first: float, second: float) -> float:
    return (second - first) / first * 100.0 if first else 0.0


# ---------------------------------------------------------------------------
# Example generators
# ---------------------------------------------------------------------------


def afr_examples(articles: list[dict], rng: random.Random, target: int) -> list[dict]:
    """Headline writing, one-sentence summary, and publication-date recall."""
    examples = []
    pool = articles[:]
    rng.shuffle(pool)
    for article in pool:
        if len(examples) >= target:
            break
        headline, intro, date = article["headline"], article["intro"], article["date"]
        text = article["text"]
        group = "afr:" + hashlib.md5(f"{date}|{headline}".encode()).hexdigest()[:12]

        # `text` opens with the intro paragraph. For the summary task the intro is
        # the target, so it is stripped from the context — otherwise the answer is
        # the first line of the prompt and the model only learns to copy.
        body = text[len(intro):].strip() if intro and text.startswith(intro) else text
        variants = ["headline"]
        if intro and intro != headline and len(body) >= 200:
            variants += ["summary", "summary"]
        if intro:
            variants += ["date"]

        choice = rng.choice(variants + ["headline", "headline"])
        if choice == "headline":
            pair = make_pair(
                "Write the headline for this Australian Financial Review article.",
                [{"publication_date": date, "intro": intro[:300],
                  "text": text[:AFR_TEXT_CHARS]}],
                headline,
                group,
            )
        elif choice == "summary":
            pair = make_pair(
                "Summarise this Australian Financial Review article in one sentence.",
                [{"publication_date": date, "text": body[:AFR_TEXT_CHARS]}],
                intro[:300],
                group,
            )
        else:
            pair = make_pair(
                f"On what date did the AFR publish the article headlined \"{headline}\"?",
                [{"headline": headline, "publication_date": date, "intro": intro[:300]}],
                f"The Australian Financial Review published \"{headline}\" on {date}.",
                group,
            )
        if pair:
            examples.append(pair)
    return examples


def asx_examples(by_ticker: dict[str, list[dict]], rng: random.Random, target: int) -> list[dict]:
    """Single-session lookups, two-date moves, and window extremes."""
    examples = []
    tickers = sorted(by_ticker)
    if not tickers:
        return examples
    attempts = 0
    while len(examples) < target and attempts < target * 6:
        attempts += 1
        ticker = rng.choice(tickers)
        rows = by_ticker[ticker]
        index = rng.randrange(len(rows))
        row = rows[index]
        group = f"asx:{ticker}:{row['date'][:4]}"
        kind = rng.random()

        if kind < 0.3:
            pair = make_pair(
                f"What was {ticker}'s closing price on {row['date']}?",
                [row],
                f"{ticker} closed at {money(row['close'])} on {row['date']}.",
                group,
            )
        elif kind < 0.45:
            pair = make_pair(
                f"What were {ticker}'s intraday high and low on {row['date']}?",
                [row],
                f"{ticker} traded between {money(row['low'])} and {money(row['high'])} "
                f"on {row['date']}, opening at {money(row['open'])} and closing at "
                f"{money(row['close'])}.",
                group,
            )
        elif kind < 0.55:
            pair = make_pair(
                f"How many {ticker} shares changed hands on {row['date']}?",
                [row],
                f"{ticker} traded {row['volume']:,} shares on {row['date']}, "
                f"closing at {money(row['close'])}.",
                group,
            )
        elif kind < 0.8:
            span = rng.choice([1, 5, 21, 63, 126])
            other = index + span
            if other >= len(rows):
                continue
            first, second = row, rows[other]
            delta = second["close"] - first["close"]
            change = pct_change(first["close"], second["close"])
            direction = "rose" if delta > 0 else "fell" if delta < 0 else "was unchanged"
            pair = make_pair(
                f"How did {ticker}'s closing price change between {first['date']} "
                f"and {second['date']}?",
                [first, second],
                f"{ticker} closed at {money(first['close'])} on {first['date']} and "
                f"{money(second['close'])} on {second['date']}. The close {direction} "
                f"by {money(abs(delta))} ({change:+.2f}%).",
                group,
            )
        else:
            window = rows[index : index + rng.choice([5, 6, 7])]
            if len(window) < 5:
                continue
            peak = max(window, key=lambda r: r["close"])
            trough = min(window, key=lambda r: r["close"])
            pair = make_pair(
                f"Across the {ticker} sessions supplied, which had the highest close "
                f"and which the lowest?",
                window,
                f"Of the {len(window)} sessions from {window[0]['date']} to "
                f"{window[-1]['date']}, {ticker} closed highest at {money(peak['close'])} "
                f"on {peak['date']} and lowest at {money(trough['close'])} on "
                f"{trough['date']}.",
                group,
            )
        if pair:
            examples.append(pair)
    return examples


def describe_decision(row: dict) -> str:
    bps = row["change_bps"]
    if bps == 0:
        return (
            f"The RBA left the cash rate target unchanged at "
            f"{row['cash_rate_target']:.2f}% on {row['effective_date']}."
        )
    verb = "raised" if bps > 0 else "cut"
    return (
        f"The RBA {verb} the cash rate target by {abs(bps)} basis points "
        f"({abs(row['change_pct']):.2f} percentage points) to "
        f"{row['cash_rate_target']:.2f}%, effective {row['effective_date']}."
    )


def rba_examples(decisions: list[dict], rng: random.Random, target: int) -> list[dict]:
    """Single decisions, spans between decisions, and per-year tallies."""
    examples = []
    if not decisions:
        return examples
    by_year: dict[str, list[dict]] = defaultdict(list)
    for row in decisions:
        by_year[row["effective_date"][:4]].append(row)
    years = sorted(by_year)

    seen: set[str] = set()
    attempts = 0
    while len(examples) < target and attempts < target * 8:
        attempts += 1
        kind = rng.random()

        if kind < 0.4:
            row = rng.choice(decisions)
            group = f"rba:{row['effective_date'][:4]}"
            pair = make_pair(
                f"What did the RBA decide at its {row['effective_date']} meeting?",
                [row],
                describe_decision(row),
                group,
            )
        elif kind < 0.75:
            i = rng.randrange(len(decisions))
            j = min(i + rng.choice([1, 2, 3, 6, 12]), len(decisions) - 1)
            if j <= i:
                continue
            first, second = decisions[i], decisions[j]
            window = decisions[i : j + 1]
            delta = second["cash_rate_target"] - first["cash_rate_target"]
            moves = [d for d in window[1:] if d["change_bps"] != 0]
            direction = "higher" if delta > 0 else "lower" if delta < 0 else "unchanged"
            pair = make_pair(
                f"How did the cash rate target move between {first['effective_date']} "
                f"and {second['effective_date']}?",
                window,
                f"The target went from {first['cash_rate_target']:.2f}% on "
                f"{first['effective_date']} to {second['cash_rate_target']:.2f}% on "
                f"{second['effective_date']} — {abs(delta):.2f} percentage points "
                f"{direction}, across {len(moves)} rate change"
                f"{'' if len(moves) == 1 else 's'}.",
                group=f"rba:{first['effective_date'][:4]}",
            )
        else:
            year = rng.choice(years)
            if year in seen:
                continue
            seen.add(year)
            rows = by_year[year]
            ups = [r for r in rows if r["change_bps"] > 0]
            downs = [r for r in rows if r["change_bps"] < 0]
            if not ups and not downs:
                answer = (
                    f"The RBA made no change in {year}: the target stayed at "
                    f"{rows[-1]['cash_rate_target']:.2f}% across all {len(rows)} decisions."
                )
            else:
                answer = (
                    f"The RBA changed the cash rate target {len(ups) + len(downs)} times "
                    f"in {year} — {len(ups)} increase{'' if len(ups) == 1 else 's'} and "
                    f"{len(downs)} decrease{'' if len(downs) == 1 else 's'} across "
                    f"{len(rows)} decisions — ending the year at "
                    f"{rows[-1]['cash_rate_target']:.2f}%."
                )
            pair = make_pair(
                f"How many times did the RBA change the cash rate target in {year}, "
                f"and where did the rate end up?",
                rows,
                answer,
                group=f"rba:{year}",
            )
        if pair:
            examples.append(pair)
    return examples


def cross_examples(
    by_ticker: dict[str, list[dict]],
    decisions: list[dict],
    rng: random.Random,
    target: int,
) -> list[dict]:
    """RBA decision plus the same-day ASX session — the cross-dataset shape."""
    examples = []
    tickers = sorted(by_ticker)
    if not tickers or not decisions:
        return examples
    price_index = {
        ticker: {row["date"]: (i, row) for i, row in enumerate(rows)}
        for ticker, rows in by_ticker.items()
    }
    attempts = 0
    while len(examples) < target and attempts < target * 10:
        attempts += 1
        decision = rng.choice(decisions)
        ticker = rng.choice(tickers)
        hit = price_index[ticker].get(decision["effective_date"])
        if not hit or hit[0] == 0:
            continue
        index, row = hit
        previous = by_ticker[ticker][index - 1]
        delta = row["close"] - previous["close"]
        change = pct_change(previous["close"], row["close"])
        verb = "rose" if delta > 0 else "fell" if delta < 0 else "was flat"
        pair = make_pair(
            f"On the RBA decision date {decision['effective_date']}, how did {ticker} "
            f"close against the previous session?",
            [decision, previous, row],
            f"{describe_decision(decision)} {ticker} closed at {money(row['close'])} "
            f"that day against {money(previous['close'])} on {previous['date']} — it "
            f"{verb} {money(abs(delta))} ({change:+.2f}%).",
            group=f"asx:{ticker}:{row['date'][:4]}",
        )
        if pair:
            examples.append(pair)
    return examples


# ---------------------------------------------------------------------------
# Splitting and writing
# ---------------------------------------------------------------------------


def split_of(group: str, seed: int) -> str:
    digest = hashlib.md5(f"{seed}:{group}".encode()).hexdigest()
    position = int(digest[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, weight in SPLIT_WEIGHTS:
        cumulative += weight
        if position < cumulative:
            return name
    return SPLIT_WEIGHTS[-1][0]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {"prompt": row["prompt"], "completion": row["completion"]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data set",
                    help="directory holding the AFR/, ASX/ and RBA Rates/ folders")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--afr-share", type=float, default=0.5,
                    help="fraction of examples drawn from the AFR corpus")
    ap.add_argument("--total", type=int, default=60_000,
                    help="target example count across all three splits")
    ap.add_argument("--smoke-size", type=int, default=500,
                    help="rows written to <out-dir>/smoke for pipeline validation")
    ap.add_argument("--seed", type=int, default=1111)
    args = ap.parse_args()

    if not 0.0 <= args.afr_share <= 1.0:
        raise SystemExit(f"--afr-share must be between 0 and 1, got {args.afr_share}")

    root = Path(args.data_root)
    if not root.is_dir():
        raise SystemExit(f"data root not found: {args.data_root!r}")

    rng = random.Random(args.seed)

    print(f"reading {root}…", flush=True)
    articles = load_afr(root)
    prices = load_asx(root)
    decisions = load_rba(root)
    print(
        f"loaded {len(articles):,} AFR articles | "
        f"{sum(len(v) for v in prices.values()):,} ASX sessions across "
        f"{len(prices)} tickers | {len(decisions):,} RBA decisions"
    )
    if not (articles or prices or decisions):
        raise SystemExit(f"no usable data found under {args.data_root!r}")

    # Overshoot the target: the group hash splits roughly, not exactly, 80/10/10,
    # and oversized examples are dropped by make_pair.
    budget = int(args.total * 1.25)
    afr_target = int(budget * args.afr_share)
    remainder = budget - afr_target
    asx_target = int(remainder * 0.55)
    rba_target = int(remainder * 0.20)
    cross_target = remainder - asx_target - rba_target

    examples = (
        afr_examples(articles, rng, afr_target)
        + asx_examples(prices, rng, asx_target)
        + rba_examples(decisions, rng, rba_target)
        + cross_examples(prices, decisions, rng, cross_target)
    )

    # Deduplicate on the prompt: the RBA corpus is small enough that random
    # sampling repeats itself.
    unique: dict[str, dict] = {}
    for example in examples:
        unique.setdefault(example["prompt"], example)
    examples = list(unique.values())
    rng.shuffle(examples)

    buckets: dict[str, list[dict]] = {name: [] for name, _ in SPLIT_WEIGHTS}
    for example in examples:
        buckets[split_of(example["group"], args.seed)].append(example)

    out_dir = Path(args.out_dir)
    for name, weight in SPLIT_WEIGHTS:
        rows = buckets[name][: int(args.total * weight)]
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        headlines = sum(1 for r in rows if "Write the headline" in r["prompt"])
        print(
            f"wrote {out_dir / f'{name}.jsonl'}: {len(rows):,} rows "
            f"({len(rows) - headlines:,} analytic / {headlines:,} headline)"
        )

    if args.smoke_size > 0:
        smoke = out_dir / "smoke"
        write_jsonl(smoke / "train.jsonl", buckets["train"][: args.smoke_size])
        write_jsonl(smoke / "val.jsonl", buckets["val"][: max(1, args.smoke_size // 10)])
        print(f"wrote {smoke}/ for pipeline validation ({args.smoke_size} rows)")


if __name__ == "__main__":
    main()
