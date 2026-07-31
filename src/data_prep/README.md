# src/data_prep

Data-prep scripts for the Nemotron fine-tuning corpus. Training-only. The runtime agent's
`query_data` tool reads `data set/` directly, full precision — nothing here is wired into
live inference.

## Scripts

| Script | Does |
|---|---|
| `copy_raw_data.py` | Optional. Stages a local raw copy of `data set/{RBA Rates,ASX,AFR}` into `data/raw/`. Read-only against the source; skip this if you don't need a separate snapshot — `clean_datasets.py` reads `data set/` directly either way. |
| `clean_datasets.py` | Cleans RBA/ASX/AFR into a training corpus at `training/data/clean/`: ISO-8601 dates, snake_case keys, ASX price rounding + `clean_ticker`, RBA `change_bps`, AFR unicode/soft-hyphen normalization. Then validates that cleaning didn't shift AFR word-boundary counts vs the known reference (QBE 2021 = 369) before declaring it safe. |

## Usage

```bash
python src/data_prep/copy_raw_data.py      # optional
python src/data_prep/clean_datasets.py     # required before generating training pairs
```

Output (`data/`, `training/data/`) is gitignored — regenerate locally, never commit it.

## Rules baked into these scripts

- Never write to `data set/`. Source is read-only.
- ASX prices round to 2dp here for token efficiency — that rounding must **not** leak into
  the runtime tool; grading tolerance on closes is ±0.0001.
- RBA `change_bps = change_pct * 100` (not `* 10000` — check the math if you touch this).
- AFR cleaning is validated against a known public answer before trusting it for exact
  pattern counts (`count`/`count_by_month`/`share`). If you change the cleaning logic, rerun
  the validation step and confirm the count still matches.

See `TEAM_PLAN.md` (G1 section) and `Sonali_plan.md` for the full reasoning.
